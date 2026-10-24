#include "engine_internal.h"
#include "tiny_gltf.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/engine.h"
#include <SDL3/SDL.h>

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

#if defined(__linux__)
#  include <wayland-client.h>
#endif

namespace engine {

void GpuContext::destroy() {
    if (queue)    { wgpuQueueRelease(queue);       queue    = nullptr; }
    if (device)   { wgpuDeviceRelease(device);     device   = nullptr; }
    if (adapter)  { wgpuAdapterRelease(adapter);   adapter  = nullptr; }
    if (instance) { wgpuInstanceRelease(instance); instance = nullptr; }
}

Engine::~Engine() { shutdown(); }

void Engine::shutdown() {
    if (triangle_pipeline_) { wgpuRenderPipelineRelease(triangle_pipeline_); triangle_pipeline_ = nullptr; }
    triangle_pipeline_ready_ = false;
    if (offscreen_view_)  { wgpuTextureViewRelease(offscreen_view_);  offscreen_view_  = nullptr; }
    if (offscreen_color_) { wgpuTextureRelease(offscreen_color_);     offscreen_color_ = nullptr; }
    if (gpu_.depth_view)     { wgpuTextureViewRelease(gpu_.depth_view);     gpu_.depth_view     = nullptr; }
    if (gpu_.depth_texture)  { wgpuTextureRelease(gpu_.depth_texture);      gpu_.depth_texture  = nullptr; }
    if (gpu_.uniform_bind_group) { wgpuBindGroupRelease(gpu_.uniform_bind_group); gpu_.uniform_bind_group = nullptr; }
    if (gpu_.uniform_bgl)        { wgpuBindGroupLayoutRelease(gpu_.uniform_bgl);  gpu_.uniform_bgl        = nullptr; }
    if (gpu_.uniform_buffer)     { wgpuBufferRelease(gpu_.uniform_buffer);        gpu_.uniform_buffer     = nullptr; }
    if (gpu_.vertex_buffer)      { wgpuBufferRelease(gpu_.vertex_buffer);         gpu_.vertex_buffer      = nullptr; }
    if (gpu_.index_buffer)       { wgpuBufferRelease(gpu_.index_buffer);          gpu_.index_buffer       = nullptr; }
    if (gpu_.gbuf_albedo_view)   { wgpuTextureViewRelease(gpu_.gbuf_albedo_view);   gpu_.gbuf_albedo_view   = nullptr; }
    if (gpu_.gbuf_albedo_tex)    { wgpuTextureRelease(gpu_.gbuf_albedo_tex);        gpu_.gbuf_albedo_tex    = nullptr; }
    if (gpu_.gbuf_normal_view)   { wgpuTextureViewRelease(gpu_.gbuf_normal_view);   gpu_.gbuf_normal_view   = nullptr; }
    if (gpu_.gbuf_normal_tex)    { wgpuTextureRelease(gpu_.gbuf_normal_tex);        gpu_.gbuf_normal_tex    = nullptr; }
    if (gpu_.gbuf_material_view) { wgpuTextureViewRelease(gpu_.gbuf_material_view); gpu_.gbuf_material_view = nullptr; }
    if (gpu_.gbuf_material_tex)  { wgpuTextureRelease(gpu_.gbuf_material_tex);      gpu_.gbuf_material_tex  = nullptr; }
    if (gpu_.gbuf_sampler)       { wgpuSamplerRelease(gpu_.gbuf_sampler);           gpu_.gbuf_sampler       = nullptr; }
    if (composite_bg_)  { wgpuBindGroupRelease(composite_bg_);           composite_bg_  = nullptr; }
    if (composite_bgl_) { wgpuBindGroupLayoutRelease(composite_bgl_);    composite_bgl_ = nullptr; }
    if (!initialized_) return;
    if (surface_.surface) { wgpuSurfaceRelease(surface_.surface); surface_.surface = nullptr; }
    gpu_.destroy();
    initialized_ = false;
}

// ---------- Adapter + device helpers ----------

struct AdapterState { WGPUAdapter adapter = nullptr; bool done = false; };
struct DeviceState  { WGPUDevice  device  = nullptr; bool done = false; };

static void adapter_cb(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView msg, void* ud1, void*) {
    auto* s = static_cast<AdapterState*>(ud1);
    if (status == WGPURequestAdapterStatus_Success) s->adapter = adapter;
    else std::fprintf(stderr, "adapter failed: %.*s\n",
                      (int)msg.length, msg.data ? msg.data : "");
    s->done = true;
}

static void device_cb(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView msg, void* ud1, void*) {
    auto* s = static_cast<DeviceState*>(ud1);
    if (status == WGPURequestDeviceStatus_Success) s->device = device;
    else std::fprintf(stderr, "device failed: %.*s\n",
                      (int)msg.length, msg.data ? msg.data : "");
    s->done = true;
}

static bool wait_for_flag(WGPUInstance instance, bool* done, int timeout_ms) {
    int waited = 0;
    while (!*done) {
        wgpuInstanceProcessEvents(instance);
        if (waited >= timeout_ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        waited += 1;
    }
    return true;
}

// ---------- Init ----------

bool Engine::init(const char* app_name,
                  int enable_validation,
                  int prefer_software_device,
                  void* wayland_surface,
                  void* wayland_display,
                  uint32_t width,
                  uint32_t height,
                  std::string& err)
{
    if (initialized_) { err = "already initialized"; return false; }

    // The llvmpipe workaround from the diagnosis: set VK_ICD_FILENAMES
    // *before* Dawn touches Vulkan. The host never sees this env var.
#if defined(__linux__)
    if (prefer_software_device) {
        setenv("VK_ICD_FILENAMES",
               "/usr/share/vulkan/icd.d/lvp_icd.json", 0);
        std::fprintf(stderr, "[engine] forcing llvmpipe (lvp_icd.json)\n");
    }
#endif

    // Enable Dawn validation + backend validation so silent failures become
    // loud stderr messages.
    const char* enabled_toggles[] = { "validation", "backend_validation" };
    WGPUDawnTogglesDescriptor val_toggles = {};
    val_toggles.chain.sType      = WGPUSType_DawnTogglesDescriptor;
    val_toggles.enabledToggleCount = 2;
    val_toggles.enabledToggles     = enabled_toggles;

    WGPUInstanceDescriptor inst_desc = {};
    inst_desc.nextInChain = &val_toggles.chain;
    gpu_.instance = wgpuCreateInstance(&inst_desc);
    if (!gpu_.instance) { err = "wgpuCreateInstance failed"; return false; }

    // ---- Adapter ----
    // Disable Dawn's internal adapter blocklist. On some Intel + Mesa
    // combinations the UHD 630 is rejected by the blocklist even though
    // vulkaninfo shows the adapter as fully functional. Also force the
    // Vulkan backend: without it, Dawn can silently fall back to Null
    // (WGPUBackendType_Null == 6) when the blocklist rejects everything.
    const char* disabled_toggle = "adapter_blocklist";
    WGPUDawnTogglesDescriptor toggles = {};
    toggles.chain.sType         = WGPUSType_DawnTogglesDescriptor;
    toggles.disabledToggleCount = 1;
    toggles.disabledToggles     = &disabled_toggle;

    WGPURequestAdapterOptions opts = {};
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    opts.backendType     = WGPUBackendType_Vulkan;
    opts.nextInChain     = &toggles.chain;

    AdapterState as;
    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode      = WGPUCallbackMode_AllowSpontaneous;
    acb.callback  = adapter_cb;
    acb.userdata1 = &as;
    wgpuInstanceRequestAdapter(gpu_.instance, &opts, acb);
    if (!wait_for_flag(gpu_.instance, &as.done, 5000) || !as.adapter) {
        err = "no Vulkan adapter (check VK_ICD_FILENAMES and Mesa)";
        shutdown();
        return false;
    }
    gpu_.adapter = as.adapter;

    // ---- Device ----
    WGPUDeviceDescriptor dev_desc = {};
    dev_desc.uncapturedErrorCallbackInfo.callback =
        [](const WGPUDevice* device, WGPUErrorType type,
           WGPUStringView message, void* /*userdata1*/, void* /*userdata2*/) {
            (void)device;
            std::fprintf(stderr, "[dawn-error] type=%d: %.*s\n",
                         (int)type, (int)message.length,
                         message.data ? message.data : "");
        };
    DeviceState ds;
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode      = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback  = device_cb;
    dcb.userdata1 = &ds;
    wgpuAdapterRequestDevice(gpu_.adapter, &dev_desc, dcb);
    if (!wait_for_flag(gpu_.instance, &ds.done, 5000) || !ds.device) {
        err = "no device";
        shutdown();
        return false;
    }
    gpu_.device = ds.device;
    gpu_.queue  = wgpuDeviceGetQueue(gpu_.device);

    // ---- Offscreen render target ----
    // Path B: render into this texture, copy to buffer, save as PNG.
    // The swapchain path still exists below but is disabled until Mesa's
    // Wayland present bug is fixed upstream.
    offscreen_width_  = width;
    offscreen_height_ = height;
    offscreen_format_ = WGPUTextureFormat_BGRA8UnormSrgb;

    {
        WGPUTextureDescriptor td = {};
        td.dimension       = WGPUTextureDimension_2D;
        td.size            = { offscreen_width_, offscreen_height_, 1 };
        td.format          = offscreen_format_;
        td.mipLevelCount   = 1;
        td.sampleCount     = 1;
        td.usage           = WGPUTextureUsage_RenderAttachment |
                             WGPUTextureUsage_CopySrc;
        offscreen_color_ = wgpuDeviceCreateTexture(gpu_.device, &td);
        if (!offscreen_color_) {
            err = "offscreen texture creation failed";
            shutdown();
            return false;
        }
        offscreen_view_ = wgpuTextureCreateView(offscreen_color_, nullptr);
        if (!offscreen_view_) {
            err = "offscreen texture view creation failed";
            shutdown();
            return false;
        }
        std::fprintf(stderr, "[engine] offscreen target: %ux%u format=%d\n",
                     offscreen_width_, offscreen_height_, (int)offscreen_format_);
    }

    // ---- Depth texture ----
    {
        WGPUTextureDescriptor td = {};
        td.dimension     = WGPUTextureDimension_2D;
        td.size          = { width, height, 1 };
        td.format        = WGPUTextureFormat_Depth32Float;
        td.mipLevelCount = 1;
        td.sampleCount   = 1;
        td.usage         = WGPUTextureUsage_RenderAttachment |
                            WGPUTextureUsage_TextureBinding;
        gpu_.depth_texture = wgpuDeviceCreateTexture(gpu_.device, &td);
        if (!gpu_.depth_texture) {
            err = "depth texture creation failed";
            shutdown();
            return false;
        }
        gpu_.depth_view = wgpuTextureCreateView(gpu_.depth_texture, nullptr);
        if (!gpu_.depth_view) {
            err = "depth view creation failed";
            shutdown();
            return false;
        }
        std::fprintf(stderr, "[engine] depth target created (%ux%u Depth32Float)\n",
                     width, height);
    }

    // ---- Uniform buffer (two mat4 = 128 bytes) ----
    {
        WGPUBufferDescriptor bd = {};
        bd.size  = 192;  // mvp(64) + model(64) + material_color(16) + params(16) + pad(32)
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        gpu_.uniform_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        if (!gpu_.uniform_buffer) {
            err = "uniform buffer creation failed";
            shutdown();
            return false;
        }
    }

    // ---- Bind group layout + bind group for the uniform buffer ----
    {
        WGPUBindGroupLayoutEntry entry = {};
        entry.binding    = 0;
        entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
        entry.buffer.type            = WGPUBufferBindingType_Uniform;
        entry.buffer.hasDynamicOffset = false;
        entry.buffer.minBindingSize   = 192;

        WGPUBindGroupLayoutDescriptor bgl_desc = {};
        bgl_desc.entryCount = 1;
        bgl_desc.entries    = &entry;
        gpu_.uniform_bgl = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl_desc);
        if (!gpu_.uniform_bgl) {
            err = "bind group layout creation failed";
            shutdown();
            return false;
        }

        WGPUBindGroupEntry bg_entry = {};
        bg_entry.binding = 0;
        bg_entry.buffer  = gpu_.uniform_buffer;
        bg_entry.offset  = 0;
        bg_entry.size    = 192;

        WGPUBindGroupDescriptor bg_desc = {};
        bg_desc.layout     = gpu_.uniform_bgl;
        bg_desc.entryCount = 1;
        bg_desc.entries    = &bg_entry;
        gpu_.uniform_bind_group = wgpuDeviceCreateBindGroup(gpu_.device, &bg_desc);
        if (!gpu_.uniform_bind_group) {
            err = "bind group creation failed";
            shutdown();
            return false;
        }
        std::fprintf(stderr, "[engine] uniform buffer + bind group ready\n");
    }

    // ---- G-buffer textures (all RGBA8Unorm, TextureBinding + RenderAttachment) ----
    auto make_gbuffer_tex = [&](WGPUTexture* out_tex, WGPUTextureView* out_view,
                                const char* name) -> bool {
        WGPUTextureDescriptor td = {};
        td.dimension     = WGPUTextureDimension_2D;
        td.size          = { width, height, 1 };
        td.format        = WGPUTextureFormat_RGBA8Unorm;
        td.mipLevelCount = 1;
        td.sampleCount   = 1;
        td.usage         = WGPUTextureUsage_RenderAttachment |
                           WGPUTextureUsage_TextureBinding |
                           WGPUTextureUsage_CopySrc;
        *out_tex = wgpuDeviceCreateTexture(gpu_.device, &td);
        if (!*out_tex) { std::fprintf(stderr, "[engine] %s texture failed\n", name); return false; }
        *out_view = wgpuTextureCreateView(*out_tex, nullptr);
        if (!*out_view) { std::fprintf(stderr, "[engine] %s view failed\n", name); return false; }
        return true;
    };

    if (!make_gbuffer_tex(&gpu_.gbuf_albedo_tex,   &gpu_.gbuf_albedo_view,   "gbuf_albedo"))   { shutdown(); return false; }
    if (!make_gbuffer_tex(&gpu_.gbuf_normal_tex,   &gpu_.gbuf_normal_view,   "gbuf_normal"))   { shutdown(); return false; }
    if (!make_gbuffer_tex(&gpu_.gbuf_material_tex, &gpu_.gbuf_material_view, "gbuf_material")) { shutdown(); return false; }
    std::fprintf(stderr, "[engine] G-buffer textures created (%ux%u, RGBA8Unorm x3)\n",
                 width, height);

    // ---- Sampler for composite pass ----
    {
        WGPUSamplerDescriptor sd = {};
        sd.addressModeU = WGPUAddressMode_ClampToEdge;
        sd.addressModeV = WGPUAddressMode_ClampToEdge;
        sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.magFilter    = WGPUFilterMode_Nearest;
        sd.minFilter    = WGPUFilterMode_Nearest;
        sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        sd.maxAnisotropy = 1;   // WebGPU requires >= 1
        gpu_.gbuf_sampler = wgpuDeviceCreateSampler(gpu_.device, &sd);
        if (!gpu_.gbuf_sampler) {
            err = "gbuf sampler creation failed";
            shutdown();
            return false;
        }
    }

    // ---- HZB texture (mipmapped R32Float, downsampled min-depth pyramid) ----
    {
        uint32_t mips = 1;
        uint32_t max_dim = width > height ? width : height;
        while (max_dim > 1) { max_dim >>= 1; ++mips; }
        gpu_.hzb_mip_count = mips;
        gpu_.hzb_width     = width;
        gpu_.hzb_height    = height;

        WGPUTextureDescriptor td = {};
        td.dimension     = WGPUTextureDimension_2D;
        td.size          = { width, height, 1 };
        td.format        = WGPUTextureFormat_R32Float;
        td.mipLevelCount = mips;
        td.sampleCount   = 1;
        td.usage         = WGPUTextureUsage_StorageBinding |
                           WGPUTextureUsage_TextureBinding |
                           WGPUTextureUsage_CopySrc;
        gpu_.hzb_texture = wgpuDeviceCreateTexture(gpu_.device, &td);
        if (!gpu_.hzb_texture) {
            err = "HZB texture creation failed";
            shutdown();
            return false;
        }

        gpu_.hzb_mip_views.resize(mips, nullptr);
        gpu_.hzb_storage_views.resize(mips, nullptr);
        for (uint32_t i = 0; i < mips; ++i) {
            WGPUTextureViewDescriptor vd = {};
            vd.format          = WGPUTextureFormat_R32Float;
            vd.dimension       = WGPUTextureViewDimension_2D;
            vd.baseMipLevel    = i;
            vd.mipLevelCount   = 1;
            vd.baseArrayLayer  = 0;
            vd.arrayLayerCount = 1;
            vd.aspect          = WGPUTextureAspect_All;
            gpu_.hzb_mip_views[i]     = wgpuTextureCreateView(gpu_.hzb_texture, &vd);
            gpu_.hzb_storage_views[i] = wgpuTextureCreateView(gpu_.hzb_texture, &vd);
            if (!gpu_.hzb_mip_views[i] || !gpu_.hzb_storage_views[i]) {
                err = "HZB mip view creation failed";
                shutdown();
                return false;
            }
        }
        std::fprintf(stderr, "[engine] HZB texture created (%ux%u, R32Float, %u mips)\n",
                     width, height, mips);
    }

    // ---- HZB convert BGL: depth input + R32Float storage output ----
    {
        WGPUBindGroupLayoutEntry e[2] = {};
        e[0].binding = 0;
        e[0].visibility = WGPUShaderStage_Compute;
        e[0].texture.sampleType = WGPUTextureSampleType_Depth;
        e[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        e[1].binding = 1;
        e[1].visibility = WGPUShaderStage_Compute;
        e[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        e[1].storageTexture.format = WGPUTextureFormat_R32Float;
        e[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bgl = {};
        bgl.entryCount = 2;
        bgl.entries    = e;
        hzb_convert_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl);
        if (!hzb_convert_bgl_) {
            err = "HZB convert BGL creation failed";
            shutdown();
            return false;
        }
    }

    // ---- HZB downsample BGL: R32Float input + R32Float storage output ----
    {
        WGPUBindGroupLayoutEntry e[2] = {};
        e[0].binding = 0;
        e[0].visibility = WGPUShaderStage_Compute;
        e[0].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        e[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        e[1].binding = 1;
        e[1].visibility = WGPUShaderStage_Compute;
        e[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        e[1].storageTexture.format = WGPUTextureFormat_R32Float;
        e[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bgl = {};
        bgl.entryCount = 2;
        bgl.entries    = e;
        hzb_downsample_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl);
        if (!hzb_downsample_bgl_) {
            err = "HZB downsample BGL creation failed";
            shutdown();
            return false;
        }
    }

    // ---- Composite bind group layout (3 textures + 1 sampler at group 1) ----
    {
        WGPUBindGroupLayoutEntry entries[4] = {};
        entries[0].binding    = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].texture.sampleType = WGPUTextureSampleType_Float;
        entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[1].binding    = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[2].binding    = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].texture.sampleType = WGPUTextureSampleType_Float;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[3].binding    = 3;
        entries[3].visibility = WGPUShaderStage_Fragment;
        entries[3].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bgl_desc = {};
        bgl_desc.entryCount = 4;
        bgl_desc.entries    = entries;
        composite_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl_desc);
        if (!composite_bgl_) {
            err = "composite BGL creation failed";
            shutdown();
            return false;
        }

        WGPUBindGroupEntry bg_entries[4] = {};
        bg_entries[0].binding     = 0;
        bg_entries[0].textureView = gpu_.gbuf_albedo_view;
        bg_entries[1].binding     = 1;
        bg_entries[1].textureView = gpu_.gbuf_normal_view;
        bg_entries[2].binding     = 2;
        bg_entries[2].textureView = gpu_.gbuf_material_view;
        bg_entries[3].binding     = 3;
        bg_entries[3].sampler     = gpu_.gbuf_sampler;

        WGPUBindGroupDescriptor bg_desc = {};
        bg_desc.layout     = composite_bgl_;
        bg_desc.entryCount = 4;
        bg_desc.entries    = bg_entries;
        composite_bg_ = wgpuDeviceCreateBindGroup(gpu_.device, &bg_desc);
        if (!composite_bg_) {
            err = "composite bind group creation failed";
            shutdown();
            return false;
        }
        std::fprintf(stderr, "[engine] composite bind group ready\n");
    }

    // ---- Cube geometry ----
    // Vertex layout: pos(vec3) + normal(vec3) + uv(vec2) = 32 bytes.
    // 24 verts (4 per face so normal/uv can be per-face), 36 indices (u16).
    {
        struct V { float px,py,pz, nx,ny,nz, u,v; };
        // Half-extent 0.5, centered at origin.
        static const V verts[24] = {
            // +X face
            { 0.5f,-0.5f,-0.5f,  1,0,0,  0,1 }, { 0.5f,-0.5f, 0.5f,  1,0,0,  1,1 },
            { 0.5f, 0.5f, 0.5f,  1,0,0,  1,0 }, { 0.5f, 0.5f,-0.5f,  1,0,0,  0,0 },
            // -X face
            {-0.5f,-0.5f, 0.5f, -1,0,0,  0,1 }, {-0.5f,-0.5f,-0.5f, -1,0,0,  1,1 },
            {-0.5f, 0.5f,-0.5f, -1,0,0,  1,0 }, {-0.5f, 0.5f, 0.5f, -1,0,0,  0,0 },
            // +Y face
            {-0.5f, 0.5f,-0.5f,  0,1,0,  0,1 }, { 0.5f, 0.5f,-0.5f,  0,1,0,  1,1 },
            { 0.5f, 0.5f, 0.5f,  0,1,0,  1,0 }, {-0.5f, 0.5f, 0.5f,  0,1,0,  0,0 },
            // -Y face
            {-0.5f,-0.5f, 0.5f,  0,-1,0, 0,1 }, { 0.5f,-0.5f, 0.5f,  0,-1,0, 1,1 },
            { 0.5f,-0.5f,-0.5f,  0,-1,0, 1,0 }, {-0.5f,-0.5f,-0.5f,  0,-1,0, 0,0 },
            // +Z face
            {-0.5f,-0.5f, 0.5f,  0,0,1,  0,1 }, { 0.5f,-0.5f, 0.5f,  0,0,1,  1,1 },
            { 0.5f, 0.5f, 0.5f,  0,0,1,  1,0 }, {-0.5f, 0.5f, 0.5f,  0,0,1,  0,0 },
            // -Z face
            { 0.5f,-0.5f,-0.5f,  0,0,-1, 0,1 }, {-0.5f,-0.5f,-0.5f,  0,0,-1, 1,1 },
            {-0.5f, 0.5f,-0.5f,  0,0,-1, 1,0 }, { 0.5f, 0.5f,-0.5f,  0,0,-1, 0,0 },
        };
        static const uint16_t idx[36] = {
             0, 1, 2,  0, 2, 3,   4, 5, 6,  4, 6, 7,
             8, 9,10,  8,10,11,  12,13,14, 12,14,15,
            16,17,18, 16,18,19,  20,21,22, 20,22,23,
        };

        WGPUBufferDescriptor vbd = {};
        vbd.size  = sizeof(verts);
        vbd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        gpu_.vertex_buffer = wgpuDeviceCreateBuffer(gpu_.device, &vbd);
        if (!gpu_.vertex_buffer) {
            err = "vertex buffer creation failed";
            shutdown();
            return false;
        }
        wgpuQueueWriteBuffer(gpu_.queue, gpu_.vertex_buffer, 0, verts, sizeof(verts));
        gpu_.vertex_count = 24;

        WGPUBufferDescriptor ibd = {};
        ibd.size  = sizeof(idx);
        ibd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
        gpu_.index_buffer = wgpuDeviceCreateBuffer(gpu_.device, &ibd);
        if (!gpu_.index_buffer) {
            err = "index buffer creation failed";
            shutdown();
            return false;
        }
        wgpuQueueWriteBuffer(gpu_.queue, gpu_.index_buffer, 0, idx, sizeof(idx));
        gpu_.index_count = 36;

        std::fprintf(stderr,
            "[engine] cube geometry uploaded: %u verts, %u indices\n",
            gpu_.vertex_count, gpu_.index_count);
    }

    // ---- Surface (Wayland only for now) ----
#if defined(__linux__)
    if (wayland_surface && wayland_display) {
        WGPUSurfaceSourceWaylandSurface src = {};
        src.chain.sType  = WGPUSType_SurfaceSourceWaylandSurface;
        src.surface      = (struct wl_surface*)wayland_surface;
        src.display      = (struct wl_display*)wayland_display;
        WGPUSurfaceDescriptor sd = {};
        sd.nextInChain = &src.chain;
        surface_.surface = wgpuInstanceCreateSurface(gpu_.instance, &sd);
        if (!surface_.surface) { err = "surface create failed"; shutdown(); return false; }
    } else {
        err = "no wayland surface provided (X11/Win32/Cocoa: to be added)";
        shutdown();
        return false;
    }
#else
    err = "only Wayland supported in this build";
    shutdown();
    return false;
#endif

    // ---- Surface configuration ----
    WGPUSurfaceCapabilities caps = {};
    wgpuSurfaceGetCapabilities(surface_.surface, gpu_.adapter, &caps);
    std::fprintf(stderr,
        "[engine] surface caps: formatCount=%zu presentModeCount=%zu "
        "alphaModeCount=%zu\n",
        caps.formatCount, caps.presentModeCount, caps.alphaModeCount);
    if (caps.formatCount == 0) {
        err = "surface reports zero supported formats";
        shutdown();
        return false;
    }
    surface_.format = caps.formats[0];
    surface_.width  = width;
    surface_.height = height;

    WGPUSurfaceConfiguration cfg = {};
    cfg.device      = gpu_.device;
    cfg.format      = surface_.format;
    cfg.usage       = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    cfg.width       = width;
    cfg.height      = height;
    cfg.presentMode = WGPUPresentMode_Fifo;
    cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(surface_.surface, &cfg);
    std::fprintf(stderr,
        "[engine] surface configured: format=%d width=%u height=%u "
        "presentMode=%d alphaMode=%d\n",
        (int)surface_.format, surface_.width, surface_.height,
        (int)cfg.presentMode, (int)cfg.alphaMode);

    initialized_ = true;
    return true;
}

void Engine::resize(uint32_t width, uint32_t height) {
    if (!initialized_ || !surface_.surface) return;
    surface_.width  = width;
    surface_.height = height;
    WGPUSurfaceConfiguration cfg = {};
    cfg.device      = gpu_.device;
    cfg.format      = surface_.format;
    cfg.usage       = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    cfg.width       = width;
    cfg.height      = height;
    cfg.presentMode = WGPUPresentMode_Fifo;
    cfg.alphaMode   = WGPUCompositeAlphaMode_Auto;
    wgpuSurfaceConfigure(surface_.surface, &cfg);
}

static WGPURenderPipeline make_geometry_pipeline(
    WGPUDevice device, WGPUTextureFormat /*unused_color_format*/,
    WGPUBindGroupLayout uniform_bgl)
{
    const char* wgsl = R"(

        struct Uniforms {
            mvp            : mat4x4<f32>,
            model          : mat4x4<f32>,
            material_color : vec4<f32>,
            material_params : vec4<f32>,  // x=roughness, y=metallic, z=ao
        };
        @group(0) @binding(0) var<uniform> u : Uniforms;

        struct VSIn {
            @location(0) pos : vec3<f32>,
            @location(1) normal : vec3<f32>,
            @location(2) uv : vec2<f32>,
        };
        struct VSOut {
            @builtin(position) clip : vec4<f32>,
            @location(0) normal : vec3<f32>,
            @location(1) uv : vec2<f32>,
        };

        @vertex
        fn vs_main(in : VSIn) -> VSOut {
            var out : VSOut;
            out.clip   = u.mvp * vec4<f32>(in.pos, 1.0);
            out.normal = in.normal;
            out.uv     = in.uv;
            return out;
        }

        fn oct_encode(n: vec3<f32>) -> vec2<f32> {
            let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-6);
            return select(p, (1.0 - abs(p.yx)) * sign(p), n.z < 0.0);
        }

        struct FSOut {
            @location(0) albedo   : vec4<f32>,
            @location(1) normal   : vec4<f32>,
            @location(2) material : vec4<f32>,
        };

        @fragment
        fn fs_main(in : VSOut) -> FSOut {
            var out : FSOut;
            let n  = normalize(in.normal);
            let oe = oct_encode(n);
            out.albedo   = u.material_color;
            out.normal   = vec4<f32>(oe.x, oe.y, 0.0, 1.0);
            out.material = vec4<f32>(u.material_params.x,
                                     u.material_params.y,
                                     u.material_params.z,
                                     1.0);
            return out;
        }
    )";

    WGPUShaderSourceWGSL wgsl_src = {};
    wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_src.code.data   = wgsl;
    wgsl_src.code.length = strlen(wgsl);

    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl_src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) { std::fprintf(stderr, "[engine] geometry shader failed\n"); return nullptr; }

    WGPUColorTargetState cts[3] = {};
    for (int i = 0; i < 3; ++i) {
        cts[i].format    = WGPUTextureFormat_RGBA8Unorm;
        cts[i].writeMask = WGPUColorWriteMask_All;
    }

    WGPUFragmentState fs = {};
    fs.module      = mod;
    fs.entryPoint.data   = "fs_main";
    fs.entryPoint.length = 7;
    fs.targetCount = 3;
    fs.targets     = cts;

    WGPUVertexAttribute attrs[3] = {};
    attrs[0].format = WGPUVertexFormat_Float32x3; attrs[0].offset = 0;  attrs[0].shaderLocation = 0;
    attrs[1].format = WGPUVertexFormat_Float32x3; attrs[1].offset = 12; attrs[1].shaderLocation = 1;
    attrs[2].format = WGPUVertexFormat_Float32x2; attrs[2].offset = 24; attrs[2].shaderLocation = 2;

    WGPUVertexBufferLayout vbl = {};
    vbl.stepMode = WGPUVertexStepMode_Vertex;
    vbl.arrayStride = 32;
    vbl.attributeCount = 3;
    vbl.attributes = attrs;

    WGPUVertexState vs = {};
    vs.module = mod;
    vs.entryPoint.data = "vs_main";
    vs.entryPoint.length = 7;
    vs.bufferCount = 1;
    vs.buffers = &vbl;

    WGPUPrimitiveState prim = {};
    prim.topology = WGPUPrimitiveTopology_TriangleList;
    prim.frontFace = WGPUFrontFace_CCW;
    prim.cullMode = WGPUCullMode_None;

    WGPUMultisampleState ms = {};
    ms.count = 1;
    ms.mask  = 0xFFFFFFFF;

    WGPUDepthStencilState ds = {};
    ds.format = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = WGPUOptionalBool_True;
    ds.depthCompare = WGPUCompareFunction_Less;
    ds.stencilFront.compare = WGPUCompareFunction_Always;
    ds.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPUPipelineLayoutDescriptor pl_desc = {};
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts = &uniform_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);
    if (!pl) { std::fprintf(stderr, "[engine] geometry pipeline layout failed\n"); wgpuShaderModuleRelease(mod); return nullptr; }

    WGPURenderPipelineDescriptor pd = {};
    pd.layout = pl;
    pd.vertex = vs;
    pd.primitive = prim;
    pd.multisample = ms;
    pd.fragment = &fs;
    pd.depthStencil = &ds;

    WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(device, &pd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] geometry pipeline: %p\n", (void*)pipe);
    return pipe;
}

static WGPURenderPipeline make_composite_pipeline(
    WGPUDevice device, WGPUTextureFormat color_format,
    WGPUBindGroupLayout uniform_bgl, WGPUBindGroupLayout gbuf_bgl)
{
    const char* wgsl = R"(
        @group(1) @binding(0) var t_albedo   : texture_2d<f32>;
        @group(1) @binding(1) var t_normal   : texture_2d<f32>;
        @group(1) @binding(2) var t_material : texture_2d<f32>;
        @group(1) @binding(3) var s_gbuf     : sampler;

        struct VSOut {
            @builtin(position) clip : vec4<f32>,
            @location(0) uv : vec2<f32>,
        };

        @vertex
        fn vs_main(@builtin(vertex_index) i : u32) -> VSOut {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 3.0, -1.0),
                vec2<f32>(-1.0,  3.0)
            );
            var out : VSOut;
            out.clip = vec4<f32>(pos[i], 0.0, 1.0);
            out.uv   = vec2<f32>((pos[i].x + 1.0) * 0.5, 1.0 - (pos[i].y + 1.0) * 0.5);
            return out;
        }

        fn oct_decode(e: vec2<f32>) -> vec3<f32> {
            let n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
            let t = clamp(-n.z, 0.0, 1.0);
            return normalize(vec3<f32>(
                n.xy + select(vec2<f32>(t), -vec2<f32>(t), n.xy >= vec2<f32>(0.0)),
                n.z));
        }

        // ---- Cook-Torrance BRDF ----
        const PI : f32 = 3.14159265359;

        // Trowbridge-Reitz / GGX normal distribution function.
        fn D_GGX(NdotH : f32, roughness : f32) -> f32 {
            let a  = roughness * roughness;
            let a2 = a * a;
            let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
            return a2 / (PI * d * d + 1e-6);
        }

        // Schlick-GGX geometry (Smith's method, single term).
        fn G_SchlickGGX(NdotV : f32, roughness : f32) -> f32 {
            let r = roughness + 1.0;
            let k = (r * r) / 8.0;
            return NdotV / (NdotV * (1.0 - k) + k + 1e-6);
        }
        fn G_Smith(NdotV : f32, NdotL : f32, roughness : f32) -> f32 {
            return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
        }

        // Fresnel-Schlick approximation.
        fn F_Schlick(F0 : vec3<f32>, cosTheta : f32) -> vec3<f32> {
            return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - cosTheta, 5.0);
        }

        @fragment
        fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
            let albedo_sample   = textureSample(t_albedo,   s_gbuf, in.uv);
            let normal_sample   = textureSample(t_normal,   s_gbuf, in.uv);
            let material_sample = textureSample(t_material, s_gbuf, in.uv);

            // Skip shading where nothing was drawn (depth-test failed pixels
            // still have our clear color; use a rough heuristic).
            if (albedo_sample.a < 0.01) {
                return vec4<f32>(0.05, 0.05, 0.1, 1.0);
            }

            let albedo    = albedo_sample.rgb;
            let N         = oct_decode(normal_sample.rg);
            let roughness = clamp(material_sample.r, 0.04, 1.0);
            let metallic  = clamp(material_sample.g, 0.0, 1.0);
            let ao        = clamp(material_sample.b, 0.0, 1.0);

            // View direction: camera at (2.5, 2.5, 2.5) looking at origin.
            // Reconstruct V in world space (approximation — not per-pixel).
            let V = normalize(vec3<f32>(2.5, 2.5, 2.5));

            // Single directional light.
            let L       = normalize(vec3<f32>(0.4, 0.8, 0.3));
            let radiance = vec3<f32>(1.0, 0.95, 0.85) * 3.0;  // warm sun

            let H     = normalize(V + L);
            let NdotL = max(dot(N, L), 0.0);
            let NdotV = max(dot(N, V), 0.001);
            let NdotH = max(dot(N, H), 0.0);
            let VdotH = max(dot(V, H), 0.0);

            // F0: dielectric (0.04) mixed toward albedo by metallic.
            let F0 = mix(vec3<f32>(0.04), albedo, metallic);

            // Specular term: Cook-Torrance.
            let D = D_GGX(NdotH, roughness);
            let G = G_Smith(NdotV, NdotL, roughness);
            let F = F_Schlick(F0, VdotH);
            let specular = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);

            // Diffuse term: Lambert, energy-conserving (metals have no diffuse).
            let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
            let diffuse = kD * albedo / PI;

            let Lo = (diffuse + specular) * radiance * NdotL;

            // Ambient: sky + ground, weighted by AO.
            let up     = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
            let sky    = mix(vec3<f32>(0.20, 0.22, 0.28),
                             vec3<f32>(0.40, 0.55, 0.85), up);
            let ground = vec3<f32>(0.10, 0.08, 0.06);
            let ambient_irradiance = mix(ground, sky, up) * ao;
            let kD_ambient = (vec3<f32>(1.0) - F_Schlick(F0, NdotV)) * (1.0 - metallic);
            let ambient = kD_ambient * albedo * ambient_irradiance;

            let color = Lo + ambient;

            // Tone mapping (ACES-ish) then sRGB handled by the swapchain.
            let x = color * 1.0;
            let tonemapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
            return vec4<f32>(clamp(tonemapped, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
        }
    )";

    WGPUShaderSourceWGSL wgsl_src = {};
    wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_src.code.data   = wgsl;
    wgsl_src.code.length = strlen(wgsl);

    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl_src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) { std::fprintf(stderr, "[engine] PBR composite shader failed\n"); return nullptr; }

    WGPUColorTargetState ct = {};
    ct.format    = color_format;
    ct.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fs = {};
    fs.module      = mod;
    fs.entryPoint.data   = "fs_main";
    fs.entryPoint.length = 7;
    fs.targetCount = 1;
    fs.targets     = &ct;

    WGPUVertexState vs = {};
    vs.module      = mod;
    vs.entryPoint.data   = "vs_main";
    vs.entryPoint.length = 7;
    vs.bufferCount = 0;
    vs.buffers     = nullptr;

    WGPUPrimitiveState prim = {};
    prim.topology  = WGPUPrimitiveTopology_TriangleList;
    prim.frontFace = WGPUFrontFace_CCW;
    prim.cullMode  = WGPUCullMode_None;

    WGPUMultisampleState ms = {};
    ms.count = 1;
    ms.mask  = 0xFFFFFFFF;

    WGPUBindGroupLayout layouts[2] = { uniform_bgl, gbuf_bgl };
    WGPUPipelineLayoutDescriptor pl_desc = {};
    pl_desc.bindGroupLayoutCount = 2;
    pl_desc.bindGroupLayouts     = layouts;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);
    if (!pl) { std::fprintf(stderr, "[engine] composite layout failed\n"); wgpuShaderModuleRelease(mod); return nullptr; }

    WGPURenderPipelineDescriptor pd = {};
    pd.layout       = pl;
    pd.vertex       = vs;
    pd.primitive    = prim;
    pd.multisample  = ms;
    pd.fragment     = &fs;
    pd.depthStencil = nullptr;

    WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(device, &pd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] PBR composite pipeline: %p\n", (void*)pipe);
    return pipe;
}




#include <vector>
#include <cstdio>

static WGPUComputePipeline make_hzb_convert_pipeline(
    WGPUDevice device, WGPUBindGroupLayout bgl)
{
    const char* wgsl = R"(
        @group(0) @binding(0) var src_depth : texture_depth_2d;
        @group(0) @binding(1) var dst       : texture_storage_2d<r32float, write>;

        @compute @workgroup_size(8, 8, 1)
        fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let dims = textureDimensions(src_depth);
            if (gid.x >= dims.x || gid.y >= dims.y) { return; }
            let d = textureLoad(src_depth, vec2<i32>(gid.xy), 0);
            textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
        }
    )";
    WGPUShaderSourceWGSL src = {};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code.data = wgsl; src.code.length = strlen(wgsl);
    WGPUShaderModuleDescriptor smd = {}; smd.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) return nullptr;

    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pld);

    WGPUComputePipelineDescriptor cpd = {};
    cpd.layout = pl;
    cpd.compute.module = mod;
    cpd.compute.entryPoint.data = "cs_main";
    cpd.compute.entryPoint.length = 7;

    WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(device, &cpd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] HZB convert pipeline: %p\n", (void*)pipe);
    return pipe;
}

static WGPUComputePipeline make_hzb_downsample_pipeline(
    WGPUDevice device, WGPUBindGroupLayout bgl)
{
    const char* wgsl = R"(
        @group(0) @binding(0) var src : texture_2d<f32>;
        @group(0) @binding(1) var dst : texture_storage_2d<r32float, write>;

        @compute @workgroup_size(8, 8, 1)
        fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let dims = textureDimensions(dst);
            if (gid.x >= dims.x || gid.y >= dims.y) { return; }
            let s = vec2<i32>(gid.xy) * 2;
            let sd = vec2<i32>(textureDimensions(src));
            let t00 = textureLoad(src, min(s + vec2<i32>(0, 0), sd - 1), 0).r;
            let t10 = textureLoad(src, min(s + vec2<i32>(1, 0), sd - 1), 0).r;
            let t01 = textureLoad(src, min(s + vec2<i32>(0, 1), sd - 1), 0).r;
            let t11 = textureLoad(src, min(s + vec2<i32>(1, 1), sd - 1), 0).r;
            let m = min(min(t00, t10), min(t01, t11));
            textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
        }
    )";
    WGPUShaderSourceWGSL src = {};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code.data = wgsl; src.code.length = strlen(wgsl);
    WGPUShaderModuleDescriptor smd = {}; smd.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) return nullptr;

    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pld);

    WGPUComputePipelineDescriptor cpd = {};
    cpd.layout = pl;
    cpd.compute.module = mod;
    cpd.compute.entryPoint.data = "cs_main";
    cpd.compute.entryPoint.length = 7;

    WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(device, &cpd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] HZB downsample pipeline: %p\n", (void*)pipe);
    return pipe;
}

void Engine::begin_frame() {
    if (!initialized_ || !offscreen_view_ || !gpu_.depth_view) return;

    // Lazy pipeline creation (outside the graph — pipelines are persistent).
    if (!geometry_pipeline_ready_) {
        geometry_pipeline_ = make_geometry_pipeline(
            gpu_.device, offscreen_format_, gpu_.uniform_bgl);
        geometry_pipeline_ready_ = true;
        if (!geometry_pipeline_) return;
    }
    if (!composite_pipeline_ready_ && composite_bgl_) {
        composite_pipeline_ = make_composite_pipeline(
            gpu_.device, offscreen_format_, gpu_.uniform_bgl, composite_bgl_);
        composite_pipeline_ready_ = true;
        if (!composite_pipeline_) return;
    }
    if (!geometry_pipeline_ || !composite_pipeline_) return;

    // Lazy HZB pipelines + bind groups.
    if (!hzb_pipelines_ready_ && hzb_convert_bgl_ && hzb_downsample_bgl_) {
        hzb_convert_pipeline_    = make_hzb_convert_pipeline(gpu_.device, hzb_convert_bgl_);
        hzb_downsample_pipeline_ = make_hzb_downsample_pipeline(gpu_.device, hzb_downsample_bgl_);
        if (hzb_convert_pipeline_ && hzb_downsample_pipeline_) {
            // Convert BG: depth_view -> hzb_mip_views[0] storage
            {
                WGPUBindGroupEntry be[2] = {};
                be[0].binding     = 0;
                be[0].textureView = gpu_.depth_view;
                be[1].binding     = 1;
                be[1].textureView = gpu_.hzb_storage_views[0];
                WGPUBindGroupDescriptor bgd = {};
                bgd.layout = hzb_convert_bgl_;
                bgd.entryCount = 2;
                bgd.entries = be;
                hzb_convert_bg_ = wgpuDeviceCreateBindGroup(gpu_.device, &bgd);
            }
            // Downsample BGs: mipN-1 (sampled) -> mipN (storage)
            hzb_downsample_bgs_.resize(gpu_.hzb_mip_count, nullptr);
            for (uint32_t i = 1; i < gpu_.hzb_mip_count; ++i) {
                WGPUBindGroupEntry be[2] = {};
                be[0].binding     = 0;
                be[0].textureView = gpu_.hzb_mip_views[i - 1];
                be[1].binding     = 1;
                be[1].textureView = gpu_.hzb_storage_views[i];
                WGPUBindGroupDescriptor bgd = {};
                bgd.layout = hzb_downsample_bgl_;
                bgd.entryCount = 2;
                bgd.entries = be;
                hzb_downsample_bgs_[i] = wgpuDeviceCreateBindGroup(gpu_.device, &bgd);
            }
            hzb_pipelines_ready_ = true;
        }
    }

    // ---- Camera: derive from scene AABB (or fixed fallback for no scene) ----
    float t = (float)SDL_GetTicks() / 1000.0f;
    glm::mat4 view;
    glm::vec3 cam_pos;
    float scene_radius = 1.0f;

    if (scene_.loaded) {
        glm::vec3 center = (scene_.aabb_min + scene_.aabb_max) * 0.5f;
        scene_radius = glm::length(scene_.aabb_max - scene_.aabb_min) * 0.5f;
        if (scene_radius < 0.01f) scene_radius = 1.0f;

        // Orbit slowly around the scene for a "hero shot" that shows the car.
        float orbit = t * 0.3f;
        glm::vec3 offset_dir = glm::vec3(std::cos(orbit), 0.55f, std::sin(orbit));
        cam_pos = center + offset_dir * scene_radius * 2.2f;
        view = glm::lookAt(cam_pos, center, glm::vec3(0.0f, 1.0f, 0.0f));
    } else {
        cam_pos = glm::vec3(2.5f, 2.5f, 2.5f);
        view = glm::lookAt(cam_pos, glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 1.0f, 0.0f));
    }

    float aspect = (float)offscreen_width_ / (float)offscreen_height_;
    float far_plane = std::max(100.0f, scene_radius * 20.0f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect,
                                      scene_radius * 0.01f, far_plane);
    // Note: proj[1][1] *= -1.0f is the usual WebGPU NDC flip. We tried that
    // and got an upside-down car, so the glTF export / node transform must
    // already produce a flip that cancels it. Removing the flip here.
    glm::mat4 mvp = proj * view;  // = proj * view, multiplied by obj.transform per object
    glm::mat4 mvp_proj_view = mvp;  // alias for the loop


    // ---- Frame graph setup ----
    // Resources persist across frames; register once per session.
    // First-frame guard: if we haven't registered resources yet, do it now.
    static bool resources_registered = false;
    PxResourceId r_uniform = PX_RESOURCE_INVALID;
    PxResourceId r_albedo  = PX_RESOURCE_INVALID;
    PxResourceId r_normal  = PX_RESOURCE_INVALID;
    PxResourceId r_material= PX_RESOURCE_INVALID;
    PxResourceId r_depth   = PX_RESOURCE_INVALID;
    PxResourceId r_offscreen = PX_RESOURCE_INVALID;
    if (!resources_registered) {
        fg_.clear();
        r_uniform   = fg_.register_resource("uniform_buffer",  nullptr);
        r_albedo    = fg_.register_resource("gbuf_albedo",     gpu_.gbuf_albedo_view);
        r_normal    = fg_.register_resource("gbuf_normal",     gpu_.gbuf_normal_view);
        r_material  = fg_.register_resource("gbuf_material",   gpu_.gbuf_material_view);
        r_depth     = fg_.register_resource("depth",           gpu_.depth_view);
        r_offscreen = fg_.register_resource("offscreen_color", offscreen_view_);
        resources_registered = true;
    } else {
        // Look up the cached ids (registered in same order as the block above).
        r_uniform   = 0;
        r_albedo    = 1;
        r_normal    = 2;
        r_material  = 3;
        r_depth     = 4;
        r_offscreen = 5;
    }

    fg_.reset_passes();

    // ---- Geometry pass: writes G-buffer + depth ----
    fg_.add_pass("geometry",
        /*reads=*/  { r_uniform },
        /*writes=*/ { r_albedo, r_normal, r_material, r_depth },
        [&](WGPUCommandEncoder enc) {
            WGPURenderPassColorAttachment gbuf_att[3] = {};
            WGPUTextureView gbuf_views[3] = {
                gpu_.gbuf_albedo_view, gpu_.gbuf_normal_view, gpu_.gbuf_material_view
            };
            for (int i = 0; i < 3; ++i) {
                gbuf_att[i].view       = gbuf_views[i];
                gbuf_att[i].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
                gbuf_att[i].loadOp     = WGPULoadOp_Clear;
                gbuf_att[i].storeOp    = WGPUStoreOp_Store;
                gbuf_att[i].clearValue = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            WGPURenderPassDepthStencilAttachment dep = {};
            dep.view            = gpu_.depth_view;
            dep.depthLoadOp     = WGPULoadOp_Clear;
            dep.depthStoreOp    = WGPUStoreOp_Store;
            dep.depthClearValue = 1.0f;
            dep.depthReadOnly   = false;

            WGPURenderPassDescriptor gp = {};
            gp.colorAttachmentCount   = 3;
            gp.colorAttachments       = gbuf_att;
            gp.depthStencilAttachment = &dep;

            WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &gp);
            wgpuRenderPassEncoderSetPipeline(rp, geometry_pipeline_);
            if (scene_.loaded) {
                // Per-object draw. Each object overwrites the uniform buffer
                // with its own MVP + model, then draws. Since uniforms are
                // written between draws and the queue is serialized, this is
                // safe (though not optimal — an instanced approach would do
                // one write + one draw call for all objects).
                for (const auto& obj : scene_.objects) {
                    glm::mat4 mvp_obj = mvp_proj_view * obj.transform;

                    struct Uniforms {
                        glm::mat4 mvp;              // 64
                        glm::mat4 model;            // 64
                        glm::vec4 material_color;   // 16
                        glm::vec4 material_params;  // 16 (roughness, metallic, ao, pad)
                        uint8_t   _pad[32];         // 32 (buffer is 192)
                    } u = {};
                    u.mvp  = mvp_obj;
                    u.model = obj.transform;

                    const auto& mesh = scene_.meshes[obj.mesh_index];
                    u.material_color = mesh.base_color;
                    u.material_params = glm::vec4(mesh.roughness, mesh.metallic, mesh.ao, 0.0f);

                    wgpuQueueWriteBuffer(gpu_.queue, gpu_.uniform_buffer, 0, &u, 192);
                    wgpuRenderPassEncoderSetBindGroup(rp, 0, gpu_.uniform_bind_group, 0, nullptr);
                    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, mesh.vertex_buffer, 0,
                        mesh.vertex_count * 32);
                    wgpuRenderPassEncoderSetIndexBuffer(rp, mesh.index_buffer,
                        WGPUIndexFormat_Uint32, 0, mesh.index_count * 4);
                    wgpuRenderPassEncoderDrawIndexed(rp, mesh.index_count, 1, 0, 0, 0);
                }
            }
            wgpuRenderPassEncoderEnd(rp);
            wgpuRenderPassEncoderRelease(rp);
        });

    // ---- HZB pass: depth -> mipmapped depth pyramid (compute) ----
    fg_.add_pass("hzb_build",
        /*reads=*/  { r_depth },
        /*writes=*/ { },  // HZB isn't tracked as an fg resource yet; no downstream pass needs it this frame
        [&](WGPUCommandEncoder enc) {
            if (!hzb_pipelines_ready_ || !hzb_convert_bg_) return;
            // Convert: depth -> hzb mip 0
            {
                WGPUComputePassDescriptor cpd = {};
                WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, &cpd);
                wgpuComputePassEncoderSetPipeline(cp, hzb_convert_pipeline_);
                wgpuComputePassEncoderSetBindGroup(cp, 0, hzb_convert_bg_, 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(
                    cp, (gpu_.hzb_width + 7) / 8, (gpu_.hzb_height + 7) / 8, 1);
                wgpuComputePassEncoderEnd(cp);
                wgpuComputePassEncoderRelease(cp);
            }
            // Downsample: mip N-1 -> mip N
            for (uint32_t i = 1; i < gpu_.hzb_mip_count; ++i) {
                uint32_t w = gpu_.hzb_width  >> i;
                uint32_t h = gpu_.hzb_height >> i;
                if (w == 0) w = 1;
                if (h == 0) h = 1;
                WGPUComputePassDescriptor cpd = {};
                WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(enc, &cpd);
                wgpuComputePassEncoderSetPipeline(cp, hzb_downsample_pipeline_);
                wgpuComputePassEncoderSetBindGroup(cp, 0, hzb_downsample_bgs_[i], 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(cp, (w + 7) / 8, (h + 7) / 8, 1);
                wgpuComputePassEncoderEnd(cp);
                wgpuComputePassEncoderRelease(cp);
            }
        });

    // ---- Composite pass: reads G-buffer, writes offscreen color ----
    fg_.add_pass("composite",
        /*reads=*/  { r_albedo, r_normal, r_material, r_uniform },
        /*writes=*/ { r_offscreen },
        [&](WGPUCommandEncoder enc) {
            WGPURenderPassColorAttachment catt = {};
            catt.view       = offscreen_view_;
            catt.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            catt.loadOp     = WGPULoadOp_Clear;
            catt.storeOp    = WGPUStoreOp_Store;
            catt.clearValue = {0.05f, 0.05f, 0.1f, 1.0f};

            WGPURenderPassDescriptor cp = {};
            cp.colorAttachmentCount   = 1;
            cp.colorAttachments       = &catt;
            cp.depthStencilAttachment = nullptr;

            WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &cp);
            wgpuRenderPassEncoderSetPipeline(rp, composite_pipeline_);
            wgpuRenderPassEncoderSetBindGroup(rp, 0, gpu_.uniform_bind_group, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(rp, 1, composite_bg_, 0, nullptr);
            wgpuRenderPassEncoderDraw(rp, 3, 1, 0, 0);
            wgpuRenderPassEncoderEnd(rp);
            wgpuRenderPassEncoderRelease(rp);
        });

    // ---- Execute ----
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
    fg_.execute(enc);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(gpu_.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    // ---- Present ----
    if (surface_.surface) {
        WGPUSurfaceTexture stex = {};
        wgpuSurfaceGetCurrentTexture(surface_.surface, &stex);
        if (stex.texture) {
            WGPUCommandEncoder pe = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
            WGPUTexelCopyTextureInfo src = {};
            src.texture  = offscreen_color_;
            src.mipLevel = 0;
            src.origin   = {0, 0, 0};
            WGPUTexelCopyTextureInfo dst = {};
            dst.texture  = stex.texture;
            dst.mipLevel = 0;
            dst.origin   = {0, 0, 0};
            WGPUExtent3D pextent = {offscreen_width_, offscreen_height_, 1};
            wgpuCommandEncoderCopyTextureToTexture(pe, &src, &dst, &pextent);
            WGPUCommandBuffer pc = wgpuCommandEncoderFinish(pe, nullptr);
            wgpuQueueSubmit(gpu_.queue, 1, &pc);
            wgpuCommandBufferRelease(pc);
            wgpuCommandEncoderRelease(pe);
            wgpuSurfacePresent(surface_.surface);
            wgpuTextureRelease(stex.texture);
        }
    }
}




void Engine::end_frame() {}
void Engine::tick()        { if (gpu_.instance) wgpuInstanceProcessEvents(gpu_.instance); }

bool Engine::load_gltf(const char* utf8_path, std::string& err) {
    if (!utf8_path || !utf8_path[0]) { err = "empty path"; return false; }
    if (!initialized_) { err = "engine not initialized"; return false; }

    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        load_err, load_warn;

    // Detect GLB vs ASCII glTF by extension.
    std::string path_str(utf8_path);
    bool is_glb = path_str.size() >= 4 &&
                  path_str.compare(path_str.size() - 4, 4, ".glb") == 0;

    bool ok = is_glb
        ? loader.LoadBinaryFromFile(&model, &load_err, &load_warn, path_str)
        : loader.LoadASCIIFromFile(&model, &load_err, &load_warn, path_str);

    if (!load_warn.empty())
        std::fprintf(stderr, "[engine] gltf warn: %s\n", load_warn.c_str());
    if (!ok) {
        err = "gltf load failed: " + load_err;
        std::fprintf(stderr, "[engine] gltf error: %s\n", load_err.c_str());
        return false;
    }

    std::fprintf(stderr,
        "[engine] gltf loaded: %zu meshes, %zu nodes, %zu accessors\n",
        model.meshes.size(), model.nodes.size(), model.accessors.size());

    // --- 1. Convert each glTF primitive into a Mesh (GPU buffers + AABB) ---
    // glTF meshes can have multiple primitives (one per material). For v1,
    // we flatten: each primitive becomes its own Mesh, and each node that
    // references a mesh gets one SceneObject per primitive.
    // We'll store per-glTF-mesh a list of (mesh_index, material_index).
    struct PrimRef { uint32_t mesh_index; uint32_t material_index; };
    std::vector<std::vector<PrimRef>> gltf_mesh_prims(model.meshes.size());

    glm::vec3 scene_min( 1e30f);
    glm::vec3 scene_max(-1e30f);

    for (size_t mi = 0; mi < model.meshes.size(); ++mi) {
        const auto& gmesh = model.meshes[mi];
        for (size_t pi = 0; pi < gmesh.primitives.size(); ++pi) {
            const auto& prim = gmesh.primitives[pi];

            // Position accessor is required.
            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end()) continue;
            const auto& pos_acc = model.accessors[pos_it->second];
            if (pos_acc.type != TINYGLTF_TYPE_VEC3 ||
                pos_acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                std::fprintf(stderr, "[engine] skip prim %zu/%zu: bad POSITION type\n",
                             mi, pi);
                continue;
            }

            // Optional normals and UVs.
            const float* normals_ptr = nullptr;
            const float* uvs_ptr     = nullptr;
            size_t normals_count = 0, uvs_count = 0;
            {
                auto it = prim.attributes.find("NORMAL");
                if (it != prim.attributes.end()) {
                    const auto& acc = model.accessors[it->second];
                    if (acc.type == TINYGLTF_TYPE_VEC3 &&
                        acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const auto& bv = model.bufferViews[acc.bufferView];
                        normals_ptr = reinterpret_cast<const float*>(
                            model.buffers[bv.buffer].data.data()
                            + bv.byteOffset + acc.byteOffset);
                        normals_count = acc.count;
                    }
                }
            }
            {
                auto it = prim.attributes.find("TEXCOORD_0");
                if (it != prim.attributes.end()) {
                    const auto& acc = model.accessors[it->second];
                    if (acc.type == TINYGLTF_TYPE_VEC2 &&
                        acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const auto& bv = model.bufferViews[acc.bufferView];
                        uvs_ptr = reinterpret_cast<const float*>(
                            model.buffers[bv.buffer].data.data()
                            + bv.byteOffset + acc.byteOffset);
                        uvs_count = acc.count;
                    }
                }
            }

            const auto& pos_bv = model.bufferViews[pos_acc.bufferView];
            const float* pos_ptr = reinterpret_cast<const float*>(
                model.buffers[pos_bv.buffer].data.data()
                + pos_bv.byteOffset + pos_acc.byteOffset);
            size_t vert_count = pos_acc.count;

            // Build 32-byte interleaved vertex data (pos, normal, uv).
            std::vector<float> vertex_data(vert_count * 8);
            glm::vec3 mmin(1e30f), mmax(-1e30f);
            for (size_t v = 0; v < vert_count; ++v) {
                float px = pos_ptr[v*3+0];
                float py = pos_ptr[v*3+1];
                float pz = pos_ptr[v*3+2];
                vertex_data[v*8+0] = px;
                vertex_data[v*8+1] = py;
                vertex_data[v*8+2] = pz;
                if (normals_ptr && v < normals_count) {
                    vertex_data[v*8+3] = normals_ptr[v*3+0];
                    vertex_data[v*8+4] = normals_ptr[v*3+1];
                    vertex_data[v*8+5] = normals_ptr[v*3+2];
                } else {
                    vertex_data[v*8+3] = 0.0f;
                    vertex_data[v*8+4] = 1.0f;
                    vertex_data[v*8+5] = 0.0f;
                }
                if (uvs_ptr && v < uvs_count) {
                    vertex_data[v*8+6] = uvs_ptr[v*2+0];
                    vertex_data[v*8+7] = uvs_ptr[v*2+1];
                } else {
                    vertex_data[v*8+6] = 0.0f;
                    vertex_data[v*8+7] = 0.0f;
                }
                mmin.x = std::min(mmin.x, px); mmax.x = std::max(mmax.x, px);
                mmin.y = std::min(mmin.y, py); mmax.y = std::max(mmax.y, py);
                mmin.z = std::min(mmin.z, pz); mmax.z = std::max(mmax.z, pz);
            }

            // Index buffer (16 or 32 bit depending on accessor count).
            std::vector<uint32_t> index_data;
            if (prim.indices >= 0) {
                const auto& idx_acc = model.accessors[prim.indices];
                const auto& idx_bv  = model.bufferViews[idx_acc.bufferView];
                const uint8_t* src = model.buffers[idx_bv.buffer].data.data()
                                    + idx_bv.byteOffset + idx_acc.byteOffset;
                index_data.resize(idx_acc.count);
                if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0; i < idx_acc.count; ++i)
                        index_data[i] = p16[i];
                } else if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(src);
                    for (size_t i = 0; i < idx_acc.count; ++i)
                        index_data[i] = p32[i];
                } else {
                    std::fprintf(stderr, "[engine] unsupported index type %d\n",
                                 idx_acc.componentType);
                    continue;
                }
            } else {
                // Non-indexed: generate 0..N-1
                index_data.resize(vert_count);
                for (size_t i = 0; i < vert_count; ++i) index_data[i] = (uint32_t)i;
            }

            // Create GPU buffers.
            Mesh mesh;
            {
                WGPUBufferDescriptor vbd = {};
                vbd.size  = vertex_data.size() * sizeof(float);
                vbd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
                mesh.vertex_buffer = wgpuDeviceCreateBuffer(gpu_.device, &vbd);
                wgpuQueueWriteBuffer(gpu_.queue, mesh.vertex_buffer, 0,
                                     vertex_data.data(), vbd.size);
            }
            {
                WGPUBufferDescriptor ibd = {};
                ibd.size  = index_data.size() * sizeof(uint32_t);
                ibd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
                mesh.index_buffer = wgpuDeviceCreateBuffer(gpu_.device, &ibd);
                wgpuQueueWriteBuffer(gpu_.queue, mesh.index_buffer, 0,
                                     index_data.data(), ibd.size);
            }
            mesh.vertex_count = (uint32_t)vert_count;
            mesh.index_count  = (uint32_t)index_data.size();
            mesh.aabb_min = mmin;
            mesh.aabb_max = mmax;

            // Read material.
            if (prim.material >= 0 && prim.material < (int)model.materials.size()) {
                const auto& mat = model.materials[prim.material];
                const auto& pbr = mat.pbrMetallicRoughness;
                if (pbr.baseColorFactor.size() == 4) {
                    mesh.base_color = glm::vec4(
                        (float)pbr.baseColorFactor[0],
                        (float)pbr.baseColorFactor[1],
                        (float)pbr.baseColorFactor[2],
                        (float)pbr.baseColorFactor[3]);
                }
                mesh.roughness = (float)pbr.roughnessFactor;
                mesh.metallic  = (float)pbr.metallicFactor;
            }

            scene_.meshes.push_back(mesh);

            uint32_t new_mi = (uint32_t)(scene_.meshes.size() - 1);
            gltf_mesh_prims[mi].push_back(PrimRef{new_mi, (uint32_t)prim.material});

            scene_min.x = std::min(scene_min.x, mmin.x); scene_max.x = std::max(scene_max.x, mmax.x);
            scene_min.y = std::min(scene_min.y, mmin.y); scene_max.y = std::max(scene_max.y, mmax.y);
            scene_min.z = std::min(scene_min.z, mmin.z); scene_max.z = std::max(scene_max.z, mmax.z);
        }
    }

    std::fprintf(stderr, "[engine] %zu GPU meshes uploaded\n", scene_.meshes.size());

    // --- 2. Walk nodes, produce SceneObjects (v1: root transform only) ---
    // Full hierarchy traversal (parent * child) is a follow-up; for a car
    // model exported from Blender/Maya, node transforms are usually already
    // baked or the entire car is one node.
    for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
        const auto& node = model.nodes[ni];
        if (node.mesh < 0) continue;

        // Local transform from TRS or matrix.
        glm::mat4 local(1.0f);
        if (node.matrix.size() == 16) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    local[c][r] = (float)node.matrix[r*4 + c];
        } else {
            if (node.translation.size() == 3)
                local = glm::translate(local, glm::vec3(
                    (float)node.translation[0],
                    (float)node.translation[1],
                    (float)node.translation[2]));
            if (node.rotation.size() == 4)
                local *= glm::mat4_cast(glm::quat(
                    (float)node.rotation[3],
                    (float)node.rotation[0],
                    (float)node.rotation[1],
                    (float)node.rotation[2]));
            if (node.scale.size() == 3)
                local = glm::scale(local, glm::vec3(
                    (float)node.scale[0],
                    (float)node.scale[1],
                    (float)node.scale[2]));
        }

        for (auto& pr : gltf_mesh_prims[node.mesh]) {
            SceneObject obj;
            obj.mesh_index = pr.mesh_index;
            obj.transform  = local;
            scene_.objects.push_back(obj);
        }
    }

    scene_.aabb_min = scene_min;
    scene_.aabb_max = scene_max;
    scene_.loaded   = true;

    std::fprintf(stderr,
        "[engine] scene: %zu objects, AABB (%.2f,%.2f,%.2f)..(%.2f,%.2f,%.2f)\n",
        scene_.objects.size(),
        scene_min.x, scene_min.y, scene_min.z,
        scene_max.x, scene_max.y, scene_max.z);

    return true;
}


void Engine::get_adapter_info(char* device, size_t device_cap,
                              char* desc,   size_t desc_cap,
                              uint32_t* backend, uint32_t* vendor, uint32_t* device_id)
{
    WGPUAdapterInfo info = {};
    wgpuAdapterGetInfo(gpu_.adapter, &info);
    if (device && device_cap) {
        size_t n = info.device.length < device_cap - 1 ? info.device.length : device_cap - 1;
        std::memcpy(device, info.device.data, n);
        device[n] = '\0';
    }
    if (desc && desc_cap) {
        size_t n = info.description.length < desc_cap - 1 ? info.description.length : desc_cap - 1;
        std::memcpy(desc, info.description.data, n);
        desc[n] = '\0';
    }
    if (backend)   *backend   = (uint32_t)info.backendType;
    if (vendor)    *vendor    = info.vendorID;
    if (device_id) *device_id = info.deviceID;
}

} // namespace engine
