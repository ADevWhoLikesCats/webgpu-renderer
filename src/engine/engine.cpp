#include "engine_internal.h"
#include <glm/gtc/matrix_transform.hpp>
#include "engine/engine.h"
#include <SDL3/SDL.h>

#include <cstdio>
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
        td.usage         = WGPUTextureUsage_RenderAttachment;
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
        bd.size  = 128;
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
        entry.buffer.minBindingSize   = 128;

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
        bg_entry.size    = 128;

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
    WGPUDevice device, WGPUTextureFormat color_format,
    WGPUBindGroupLayout uniform_bgl)
{
    const char* wgsl = R"(
        struct Uniforms {
            mvp : mat4x4<f32>,
            model : mat4x4<f32>,
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

        @fragment
        fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
            // Simple fake shade using normal so faces are distinguishable.
            let n = normalize(in.normal);
            let light = vec3<f32>(0.4, 0.8, 0.3);
            let d = max(dot(n, normalize(light)), 0.0);
            return vec4<f32>(vec3<f32>(1.0, 0.5, 0.1) * (0.3 + 0.7 * d), 1.0);
        }
    )";

    WGPUShaderSourceWGSL wgsl_src = {};
    wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_src.code.data   = wgsl;
    wgsl_src.code.length = strlen(wgsl);

    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &wgsl_src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) { std::fprintf(stderr, "[engine] cube shader module failed\n"); return nullptr; }

    WGPUColorTargetState ct = {};
    ct.format    = color_format;
    ct.writeMask = WGPUColorWriteMask_All;

    WGPUFragmentState fs = {};
    fs.module      = mod;
    fs.entryPoint.data   = "fs_main";
    fs.entryPoint.length = 7;
    fs.targetCount = 1;
    fs.targets     = &ct;

    WGPUVertexAttribute attrs[3] = {};
    attrs[0].format         = WGPUVertexFormat_Float32x3;
    attrs[0].offset         = 0;
    attrs[0].shaderLocation = 0;
    attrs[1].format         = WGPUVertexFormat_Float32x3;
    attrs[1].offset         = 12;
    attrs[1].shaderLocation = 1;
    attrs[2].format         = WGPUVertexFormat_Float32x2;
    attrs[2].offset         = 24;
    attrs[2].shaderLocation = 2;

    WGPUVertexBufferLayout vbl = {};
    vbl.stepMode       = WGPUVertexStepMode_Vertex;
    vbl.arrayStride    = 32;
    vbl.attributeCount = 3;
    vbl.attributes     = attrs;

    WGPUVertexState vs = {};
    vs.module      = mod;
    vs.entryPoint.data   = "vs_main";
    vs.entryPoint.length = 7;
    vs.bufferCount = 1;
    vs.buffers     = &vbl;

    WGPUPrimitiveState prim = {};
    prim.topology  = WGPUPrimitiveTopology_TriangleList;
    prim.frontFace = WGPUFrontFace_CCW;
    prim.cullMode  = WGPUCullMode_None;  // show all faces while debugging

    WGPUMultisampleState ms = {};
    ms.count = 1;
    ms.mask  = 0xFFFFFFFF;

    WGPUDepthStencilState ds = {};
    ds.format            = WGPUTextureFormat_Depth32Float;
    ds.depthWriteEnabled = WGPUOptionalBool_True;
    ds.depthCompare      = WGPUCompareFunction_Less;
    ds.stencilFront.compare = WGPUCompareFunction_Always;
    ds.stencilBack.compare  = WGPUCompareFunction_Always;

    WGPUPipelineLayoutDescriptor pl_desc = {};
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts     = &uniform_bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);
    if (!pl) { std::fprintf(stderr, "[engine] cube pipeline layout failed\n"); wgpuShaderModuleRelease(mod); return nullptr; }

    WGPURenderPipelineDescriptor pd = {};
    pd.layout      = pl;
    pd.vertex      = vs;
    pd.primitive   = prim;
    pd.multisample = ms;
    pd.fragment    = &fs;
    pd.depthStencil = &ds;

    WGPURenderPipeline pipe = wgpuDeviceCreateRenderPipeline(device, &pd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] cube pipeline: %p\n", (void*)pipe);
    return pipe;
}


#include <vector>
#include <cstdio>

void Engine::begin_frame() {
    if (!initialized_ || !offscreen_color_ || !gpu_.depth_view) return;

    if (!geometry_pipeline_ready_) {
        geometry_pipeline_ = make_geometry_pipeline(
            gpu_.device, offscreen_format_, gpu_.uniform_bgl);
        geometry_pipeline_ready_ = true;
        if (!geometry_pipeline_) { std::fprintf(stderr, "[engine] no cube pipeline\n"); return; }
    }

    // ---- Compute MVP ----
    float t = (float)SDL_GetTicks() / 1000.0f;
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), t * 0.8f, glm::vec3(0.4f, 1.0f, 0.2f));
    glm::mat4 view  = glm::lookAt(glm::vec3(2.5f, 2.5f, 2.5f),
                                  glm::vec3(0.0f, 0.0f, 0.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f));
    float aspect = (float)offscreen_width_ / (float)offscreen_height_;
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;  // WebGPU clip space: Y down
    glm::mat4 mvp = proj * view * model;

    struct { glm::mat4 mvp; glm::mat4 model; } ubo;
    ubo.mvp = mvp; ubo.model = model;
    wgpuQueueWriteBuffer(gpu_.queue, gpu_.uniform_buffer, 0, &ubo, sizeof(ubo));

    // ---- Render pass ----
    WGPURenderPassColorAttachment color = {};
    color.view       = offscreen_view_;
    color.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    color.loadOp     = WGPULoadOp_Clear;
    color.storeOp    = WGPUStoreOp_Store;
    color.clearValue = {0.05f, 0.05f, 0.1f, 1.0f};

    WGPURenderPassDepthStencilAttachment depth = {};
    depth.view            = gpu_.depth_view;
    depth.depthLoadOp     = WGPULoadOp_Clear;
    depth.depthStoreOp    = WGPUStoreOp_Store;
    depth.depthClearValue = 1.0f;
    depth.depthReadOnly   = false;

    WGPURenderPassDescriptor pass = {};
    pass.colorAttachmentCount     = 1;
    pass.colorAttachments         = &color;
    pass.depthStencilAttachment   = &depth;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
    WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(enc, &pass);
    wgpuRenderPassEncoderSetPipeline(rp, geometry_pipeline_);
    wgpuRenderPassEncoderSetBindGroup(rp, 0, gpu_.uniform_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, gpu_.vertex_buffer, 0, 24 * 32);
    wgpuRenderPassEncoderSetIndexBuffer(rp, gpu_.index_buffer,
                                        WGPUIndexFormat_Uint16, 0, 36 * 2);
    wgpuRenderPassEncoderDrawIndexed(rp, 36, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(rp);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(gpu_.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuRenderPassEncoderRelease(rp);
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

bool Engine::load_gltf(const char*, std::string& err) {
    err = "gltf loading not implemented yet";
    return false;
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
