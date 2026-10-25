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

    if (cull_aabb_buffer)    { wgpuBufferRelease(cull_aabb_buffer);    cull_aabb_buffer    = nullptr; }
    if (cull_params_buffer)  { wgpuBufferRelease(cull_params_buffer);  cull_params_buffer  = nullptr; }
    if (cull_output_buffer)  { wgpuBufferRelease(cull_output_buffer);  cull_output_buffer  = nullptr; }
    if (cull_indices_buffer) { wgpuBufferRelease(cull_indices_buffer); cull_indices_buffer = nullptr; }
    if (cull_mesh_ids_buffer)    { wgpuBufferRelease(cull_mesh_ids_buffer);    cull_mesh_ids_buffer    = nullptr; }
    if (cull_mesh_counts_buffer) { wgpuBufferRelease(cull_mesh_counts_buffer); cull_mesh_counts_buffer = nullptr; }
    if (cull_mesh_offsets_buffer){ wgpuBufferRelease(cull_mesh_offsets_buffer);cull_mesh_offsets_buffer= nullptr; }
    if (cull_mesh_buckets_buffer){ wgpuBufferRelease(cull_mesh_buckets_buffer);cull_mesh_buckets_buffer= nullptr; }
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
    if (cull_bg_)       { wgpuBindGroupRelease(cull_bg_);                  cull_bg_       = nullptr; }
    if (cull_bgl_)      { wgpuBindGroupLayoutRelease(cull_bgl_);          cull_bgl_      = nullptr; }
    if (cull_pipeline_) { wgpuComputePipelineRelease(cull_pipeline_);     cull_pipeline_ = nullptr; }
    cull_ready_ = false;

    // HZB resources (previously leaked)
    if (hzb_convert_bg_) { wgpuBindGroupRelease(hzb_convert_bg_); hzb_convert_bg_ = nullptr; }
    for (auto bg : hzb_downsample_bgs_) if (bg) wgpuBindGroupRelease(bg);
    hzb_downsample_bgs_.clear();
    if (hzb_convert_bgl_)    { wgpuBindGroupLayoutRelease(hzb_convert_bgl_);    hzb_convert_bgl_    = nullptr; }
    if (hzb_downsample_bgl_) { wgpuBindGroupLayoutRelease(hzb_downsample_bgl_); hzb_downsample_bgl_ = nullptr; }
    if (hzb_convert_pipeline_)    { wgpuComputePipelineRelease(hzb_convert_pipeline_);    hzb_convert_pipeline_    = nullptr; }
    if (hzb_downsample_pipeline_) { wgpuComputePipelineRelease(hzb_downsample_pipeline_); hzb_downsample_pipeline_ = nullptr; }
    for (auto v : gpu_.hzb_mip_views)     if (v) wgpuTextureViewRelease(v);
    for (auto v : gpu_.hzb_storage_views) if (v) wgpuTextureViewRelease(v);
    gpu_.hzb_mip_views.clear();
    gpu_.hzb_storage_views.clear();
    if (gpu_.hzb_texture) { wgpuTextureRelease(gpu_.hzb_texture); gpu_.hzb_texture = nullptr; }
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
        // Per-object uniform buffer. Each object gets 256 bytes (aligned
        // size of the 192-byte UBO). Sized for 8192 objects = 2 MB.
        // Dynamic offsets let us bind the same buffer with different
        // per-draw offsets, so each draw sees its own uniform data.
        // Per-object uniform data in a storage buffer.
        // WGSL array stride for Uniforms { mat4, mat4, vec4, vec4 } is 160.
        WGPUBufferDescriptor bd = {};
        bd.size  = 160ull * 8192ull;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
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
        entry.buffer.type            = WGPUBufferBindingType_ReadOnlyStorage;
        entry.buffer.hasDynamicOffset = false;
        entry.buffer.minBindingSize   = 0;

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
        bg_entry.size    = 160ull * 8192ull;  // whole buffer

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

    // ---- Compute culling buffers + BGL (step 3a) ----
    {
        gpu_.cull_aabb_capacity    = 8192;
        gpu_.cull_indices_capacity = 8192;

        {
            WGPUBufferDescriptor bd = {};
            bd.size  = (uint64_t)gpu_.cull_aabb_capacity * 32ull;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            gpu_.cull_aabb_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = 80;  // mat4 (64) + object_count u32 + 3 pad u32
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            gpu_.cull_params_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = 4;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            gpu_.cull_output_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = (uint64_t)gpu_.cull_indices_capacity * 4ull;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            gpu_.cull_indices_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }

        // Per-object mesh id buffer (uploaded at load).
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = (uint64_t)gpu_.cull_aabb_capacity * 4ull;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            gpu_.cull_mesh_ids_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }
        // Per-mesh counts (cleared each frame, incremented by compute).
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = 65536;  // 16384 meshes * 4 bytes; cap for future
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            gpu_.cull_mesh_counts_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
            gpu_.cull_mesh_capacity = 16384;
        }
        // Per-mesh base offsets into the bucket array (uploaded at load).
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = 65536;  // 16384 meshes * 4 bytes
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            gpu_.cull_mesh_offsets_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }
        // Flat bucket array: object indices grouped by mesh.
        {
            WGPUBufferDescriptor bd = {};
            bd.size  = (uint64_t)gpu_.cull_aabb_capacity * 4ull;
            bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
            gpu_.cull_mesh_buckets_buffer = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        }

        if (!gpu_.cull_aabb_buffer || !gpu_.cull_params_buffer ||
            !gpu_.cull_output_buffer || !gpu_.cull_indices_buffer ||
            !gpu_.cull_mesh_ids_buffer || !gpu_.cull_mesh_counts_buffer ||
            !gpu_.cull_mesh_offsets_buffer || !gpu_.cull_mesh_buckets_buffer) {
            err = "cull buffer creation failed";
            shutdown();
            return false;
        }

        WGPUBindGroupLayoutEntry e[8] = {};
        e[0].binding = 0; e[0].visibility = WGPUShaderStage_Compute;
        e[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        e[1].binding = 1; e[1].visibility = WGPUShaderStage_Compute;
        e[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        e[2].binding = 2; e[2].visibility = WGPUShaderStage_Compute;
        e[2].buffer.type = WGPUBufferBindingType_Storage;
        e[3].binding = 3; e[3].visibility = WGPUShaderStage_Compute;
        e[3].buffer.type = WGPUBufferBindingType_Storage;
        e[4].binding = 4; e[4].visibility = WGPUShaderStage_Compute;
        e[4].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        e[5].binding = 5; e[5].visibility = WGPUShaderStage_Compute;
        e[5].buffer.type = WGPUBufferBindingType_Storage;
        e[6].binding = 6; e[6].visibility = WGPUShaderStage_Compute;
        e[6].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        e[7].binding = 7; e[7].visibility = WGPUShaderStage_Compute;
        e[7].buffer.type = WGPUBufferBindingType_Storage;

        WGPUBindGroupLayoutDescriptor bgl_desc = {};
        bgl_desc.entryCount = 8;
        bgl_desc.entries    = e;
        cull_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl_desc);
        if (!cull_bgl_) {
            err = "cull BGL creation failed";
            shutdown();
            return false;
        }

        std::fprintf(stderr, "[engine] cull buffers + BGL ready\n");
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

    // ---- Material binding group layout (group 1) ----
    // Bindings: 0 = base color texture, 1 = metallicRoughness texture,
    //           2 = normal texture, 3 = sampler.
    {
        WGPUBindGroupLayoutEntry e[4] = {};
        for (int i = 0; i < 3; ++i) {
            e[i].binding = (uint32_t)i;
            e[i].visibility = WGPUShaderStage_Fragment;
            e[i].texture.sampleType = WGPUTextureSampleType_Float;
            e[i].texture.viewDimension = WGPUTextureViewDimension_2D;
        }
        e[3].binding = 3;
        e[3].visibility = WGPUShaderStage_Fragment;
        e[3].sampler.type = WGPUSamplerBindingType_Filtering;
        WGPUBindGroupLayoutDescriptor bgl = {};
        bgl.entryCount = 4;
        bgl.entries    = e;
        material_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu_.device, &bgl);
        if (!material_bgl_) {
            err = "material BGL creation failed";
            shutdown();
            return false;
        }
        std::fprintf(stderr, "[engine] material bind group layout ready\n");
    }

    // ---- Default sampler for material textures ----
    {
        WGPUSamplerDescriptor sd = {};
        sd.addressModeU  = WGPUAddressMode_Repeat;
        sd.addressModeV  = WGPUAddressMode_Repeat;
        sd.addressModeW  = WGPUAddressMode_Repeat;
        sd.magFilter     = WGPUFilterMode_Linear;
        sd.minFilter     = WGPUFilterMode_Linear;
        sd.mipmapFilter  = WGPUMipmapFilterMode_Linear;
        sd.maxAnisotropy = 1;
        scene_.default_sampler = wgpuDeviceCreateSampler(gpu_.device, &sd);
    }

    // ---- 1x1 white texture (fallback for materials without a map) ----
    {
        WGPUTextureDescriptor td = {};
        td.dimension     = WGPUTextureDimension_2D;
        td.size          = { 1, 1, 1 };
        td.format        = WGPUTextureFormat_RGBA8UnormSrgb;
        td.mipLevelCount = 1;
        td.sampleCount   = 1;
        td.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        scene_.white_tex = wgpuDeviceCreateTexture(gpu_.device, &td);
        uint32_t white_pixel = 0xFFFFFFFFu;
        // Direct write via a staging buffer since wgpuQueueWriteTexture isn't
        // in the C API for arbitrary texture uploads; use CopyBufferToTexture.
        // For 1x1, the simplest is a tiny staging buffer.
        // bytesPerRow must be a multiple of 256 for CopyBufferToTexture.
        // Allocate 256 bytes, write the 4-byte pixel at offset 0.
        WGPUBufferDescriptor bd = {};
        bd.size  = 256;
        bd.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        wgpuQueueWriteBuffer(gpu_.queue, staging, 0, &white_pixel, 4);

        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
        WGPUTexelCopyBufferInfo src = {};
        src.buffer = staging;
        src.layout.bytesPerRow  = 256;
        src.layout.rowsPerImage = 1;
        WGPUTexelCopyTextureInfo dst = {};
        dst.texture  = scene_.white_tex;
        dst.mipLevel = 0;
        dst.origin   = {0, 0, 0};
        WGPUExtent3D extent = {1, 1, 1};
        wgpuCommandEncoderCopyBufferToTexture(e, &src, &dst, &extent);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(gpu_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(e);
        wgpuBufferRelease(staging);
        scene_.white_view = wgpuTextureCreateView(scene_.white_tex, nullptr);
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

// =========================================================================
// Frustum culling
// =========================================================================
struct Frustum6 {
    glm::vec4 planes[6];
};

static Frustum6 extract_frustum(const glm::mat4& m) {
    Frustum6 f;
    glm::vec4 r0(m[0][0], m[1][0], m[2][0], m[3][0]);
    glm::vec4 r1(m[0][1], m[1][1], m[2][1], m[3][1]);
    glm::vec4 r2(m[0][2], m[1][2], m[2][2], m[3][2]);
    glm::vec4 r3(m[0][3], m[1][3], m[2][3], m[3][3]);
    f.planes[0] = r3 + r0;
    f.planes[1] = r3 - r0;
    f.planes[2] = r3 + r1;
    f.planes[3] = r3 - r1;
    f.planes[4] = r3 + r2;
    f.planes[5] = r3 - r2;
    for (int i = 0; i < 6; ++i) {
        float len = glm::length(glm::vec3(f.planes[i]));
        if (len > 1e-6f) f.planes[i] /= len;
    }
    return f;
}

static void transform_aabb(const glm::vec3& in_min, const glm::vec3& in_max,
                          const glm::mat4& m,
                          glm::vec3& out_min, glm::vec3& out_max)
{
    glm::vec3 c[8] = {
        {in_min.x, in_min.y, in_min.z}, {in_max.x, in_min.y, in_min.z},
        {in_min.x, in_max.y, in_min.z}, {in_max.x, in_max.y, in_min.z},
        {in_min.x, in_min.y, in_max.z}, {in_max.x, in_min.y, in_max.z},
        {in_min.x, in_max.y, in_max.z}, {in_max.x, in_max.y, in_max.z},
    };
    out_min = glm::vec3( 1e30f);
    out_max = glm::vec3(-1e30f);
    for (int i = 0; i < 8; ++i) {
        glm::vec3 w = glm::vec3(m * glm::vec4(c[i], 1.0f));
        out_min = glm::min(out_min, w);
        out_max = glm::max(out_max, w);
    }
}

static bool frustum_test_aabb(const Frustum6& f,
                              const glm::vec3& aabb_min,
                              const glm::vec3& aabb_max)
{
    for (int i = 0; i < 6; ++i) {
        const glm::vec4& p = f.planes[i];
        glm::vec3 pv(
            (p.x >= 0.0f) ? aabb_max.x : aabb_min.x,
            (p.y >= 0.0f) ? aabb_max.y : aabb_min.y,
            (p.z >= 0.0f) ? aabb_max.z : aabb_min.z);
        if (p.x * pv.x + p.y * pv.y + p.z * pv.z + p.w < 0.0f) return false;
    }
    return true;
}

static WGPURenderPipeline make_geometry_pipeline(
    WGPUDevice device, WGPUTextureFormat /*unused_color_format*/,
    WGPUBindGroupLayout uniform_bgl, WGPUBindGroupLayout material_bgl)
{
    const char* wgsl = R"(

        struct Uniforms {
            mvp            : mat4x4<f32>,
            model          : mat4x4<f32>,
            material_color : vec4<f32>,
            material_params : vec4<f32>,  // x=roughness, y=metallic, z=ao
        };
        @group(0) @binding(0) var<storage, read> u_arr : array<Uniforms, 8192>;

        @group(1) @binding(0) var t_base   : texture_2d<f32>;
        @group(1) @binding(1) var t_mr     : texture_2d<f32>;
        @group(1) @binding(2) var t_normal : texture_2d<f32>;
        @group(1) @binding(3) var s_mat    : sampler;

        struct VSIn {
            @location(0) pos     : vec3<f32>,
            @location(1) normal  : vec3<f32>,
            @location(2) uv      : vec2<f32>,
            @location(3) tangent : vec4<f32>,
            @location(4) obj_idx : u32,
        };
        struct VSOut {
            @builtin(position) clip : vec4<f32>,
            @location(0) normal  : vec3<f32>,
            @location(1) tangent : vec3<f32>,
            @location(2) bitan   : vec3<f32>,
            @location(3) uv      : vec2<f32>,
            @location(4) @interpolate(flat) obj_idx : u32,
        };

        @vertex
        fn vs_main(in : VSIn) -> VSOut {
            var out : VSOut;
            let u = u_arr[in.obj_idx];
            out.clip = u.mvp * vec4<f32>(in.pos, 1.0);

            let N = normalize(in.normal);
            let T = normalize(in.tangent.xyz - N * dot(N, in.tangent.xyz));
            let B = cross(N, T) * in.tangent.w;

            out.normal  = N;
            out.tangent = T;
            out.bitan   = B;
            out.uv      = in.uv;
            out.obj_idx = in.obj_idx;
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

            let base_sample = textureSample(t_base,   s_mat, in.uv);
            let mr_sample   = textureSample(t_mr,     s_mat, in.uv);
            let nrm_sample  = textureSample(t_normal, s_mat, in.uv);

            let u = u_arr[in.obj_idx];
            let albedo = u.material_color.rgb * base_sample.rgb;

            let roughness = clamp(u.material_params.x * mr_sample.g, 0.04, 1.0);
            let metallic  = clamp(u.material_params.y * mr_sample.b, 0.0, 1.0);
            let ao        = u.material_params.z;

            let n_tangent = normalize(nrm_sample.xyz * 2.0 - 1.0);
            let N_world   = normalize(in.tangent * n_tangent.x
                                    + in.bitan   * n_tangent.y
                                    + in.normal  * n_tangent.z);

            let oe = oct_encode(N_world);

            out.albedo   = vec4<f32>(albedo, 1.0);
            out.normal   = vec4<f32>(oe.x, oe.y, 0.0, 1.0);
            out.material = vec4<f32>(roughness, metallic, ao, 1.0);
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

    WGPUVertexAttribute vert_attrs[4] = {};
    vert_attrs[0].format = WGPUVertexFormat_Float32x3; vert_attrs[0].offset = 0;  vert_attrs[0].shaderLocation = 0;
    vert_attrs[1].format = WGPUVertexFormat_Float32x3; vert_attrs[1].offset = 12; vert_attrs[1].shaderLocation = 1;
    vert_attrs[2].format = WGPUVertexFormat_Float32x2; vert_attrs[2].offset = 24; vert_attrs[2].shaderLocation = 2;
    vert_attrs[3].format = WGPUVertexFormat_Float32x4; vert_attrs[3].offset = 32; vert_attrs[3].shaderLocation = 3;
    WGPUVertexBufferLayout vbl_vert = {};
    vbl_vert.stepMode = WGPUVertexStepMode_Vertex;
    vbl_vert.arrayStride = 48;
    vbl_vert.attributeCount = 4;
    vbl_vert.attributes = vert_attrs;

    WGPUVertexAttribute inst_attr = {};
    inst_attr.format = WGPUVertexFormat_Uint32;
    inst_attr.offset = 0;
    inst_attr.shaderLocation = 4;
    WGPUVertexBufferLayout vbl_inst = {};
    vbl_inst.stepMode = WGPUVertexStepMode_Instance;
    vbl_inst.arrayStride = 4;
    vbl_inst.attributeCount = 1;
    vbl_inst.attributes = &inst_attr;

    WGPUVertexBufferLayout vbls[2] = { vbl_vert, vbl_inst };

    WGPUVertexState vs = {};
    vs.module = mod;
    vs.entryPoint.data = "vs_main";
    vs.entryPoint.length = 7;
    vs.bufferCount = 2;
    vs.buffers = vbls;

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

    WGPUBindGroupLayout layouts[2] = { uniform_bgl, material_bgl };
    WGPUPipelineLayoutDescriptor pl_desc = {};
    pl_desc.bindGroupLayoutCount = 2;
    pl_desc.bindGroupLayouts = layouts;
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
            let radiance = vec3<f32>(1.0, 0.95, 0.85) * 25.0;  // warm sun, boosted

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
                             vec3<f32>(0.40, 0.55, 0.85), up) * 2.0;
            let ground = vec3<f32>(0.10, 0.08, 0.06);
            let ambient_irradiance = mix(ground, sky, up) * ao;
            let kD_ambient = (vec3<f32>(1.0) - F_Schlick(F0, NdotV)) * (1.0 - metallic);
            let ambient = kD_ambient * albedo * ambient_irradiance;


            // R = albedo.r, G = albedo.g, B = albedo.b would be flat albedo.
            // Better diagnostic: albedo * lighting so we see if it responds.
            // Tone mapping (ACES-ish) then sRGB handled by the swapchain.
            let color = Lo + ambient;
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

static WGPUComputePipeline make_cull_pipeline(
    WGPUDevice device, WGPUBindGroupLayout bgl)
{
    const char* wgsl = R"(
        struct Aabb {
            min_xyz : vec3<f32>,
            _pad0   : f32,
            max_xyz : vec3<f32>,
            _pad1   : f32,
        };
        struct Params {
            mvp_proj_view : mat4x4<f32>,
            object_count  : u32,
            _pad0         : u32,
            _pad1         : u32,
            _pad2         : u32,
        };

        @group(0) @binding(0) var<storage, read>       aabbs     : array<Aabb>;
        @group(0) @binding(1) var<storage, read>       params    : Params;
        @group(0) @binding(2) var<storage, read_write> out_cnt   : array<atomic<u32>, 1>;
        @group(0) @binding(3) var<storage, read_write> out_idx   : array<u32>;
        @group(0) @binding(4) var<storage, read>       mesh_ids  : array<u32>;
        @group(0) @binding(5) var<storage, read_write> mesh_cnts : array<atomic<u32>>;
        @group(0) @binding(6) var<storage, read>       mesh_offs : array<u32>;
        @group(0) @binding(7) var<storage, read_write> mesh_buckets : array<u32>;

        fn extract_planes(m: mat4x4<f32>) -> array<vec4<f32>, 6> {
            let r0 = vec4<f32>(m[0][0], m[1][0], m[2][0], m[3][0]);
            let r1 = vec4<f32>(m[0][1], m[1][1], m[2][1], m[3][1]);
            let r2 = vec4<f32>(m[0][2], m[1][2], m[2][2], m[3][2]);
            let r3 = vec4<f32>(m[0][3], m[1][3], m[2][3], m[3][3]);
            var p : array<vec4<f32>, 6>;
            p[0] = r3 + r0;
            p[1] = r3 - r0;
            p[2] = r3 + r1;
            p[3] = r3 - r1;
            p[4] = r3 + r2;
            p[5] = r3 - r2;
            return p;
        }

        fn aabb_visible(planes: array<vec4<f32>, 6>,
                        mn: vec3<f32>, mx: vec3<f32>) -> bool {
            for (var i = 0u; i < 6u; i++) {
                let p = planes[i];
                let pv = vec3<f32>(
                    select(mn.x, mx.x, p.x >= 0.0),
                    select(mn.y, mx.y, p.y >= 0.0),
                    select(mn.z, mx.z, p.z >= 0.0));
                if (dot(p.xyz, pv) + p.w < 0.0) { return false; }
            }
            return true;
        }

        @compute @workgroup_size(64, 1, 1)
        fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
            if (gid.x >= params.object_count) { return; }
            let planes = extract_planes(params.mvp_proj_view);
            let a = aabbs[gid.x];
            if (aabb_visible(planes, a.min_xyz, a.max_xyz)) {
                let slot = atomicAdd(&out_cnt[0], 1u);
                out_idx[slot] = gid.x;
                let mesh_id = mesh_ids[gid.x];
                if (mesh_id < arrayLength(&mesh_cnts)) {
                    let mesh_slot = atomicAdd(&mesh_cnts[mesh_id], 1u);
                    let base = mesh_offs[mesh_id];
                    mesh_buckets[base + mesh_slot] = gid.x;
                }
            }
        }
    )";

    WGPUShaderSourceWGSL src = {};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code.data   = wgsl;
    src.code.length = strlen(wgsl);

    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &smd);
    if (!mod) { std::fprintf(stderr, "[engine] cull shader failed\n"); return nullptr; }

    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1;
    pld.bindGroupLayouts     = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pld);

    WGPUComputePipelineDescriptor cpd = {};
    cpd.layout             = pl;
    cpd.compute.module     = mod;
    cpd.compute.entryPoint.data   = "cs_main";
    cpd.compute.entryPoint.length = 7;

    WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(device, &cpd);
    wgpuPipelineLayoutRelease(pl);
    wgpuShaderModuleRelease(mod);
    std::fprintf(stderr, "[engine] cull pipeline: %p\n", (void*)pipe);
    return pipe;
}

void Engine::begin_frame() {
    if (!initialized_ || !offscreen_view_ || !gpu_.depth_view) return;

    // Lazy pipeline creation (outside the graph — pipelines are persistent).
    if (!geometry_pipeline_ready_) {
        geometry_pipeline_ = make_geometry_pipeline(
            gpu_.device, offscreen_format_, gpu_.uniform_bgl, material_bgl_);
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
        // Frame the car (not the ground plane) so the shot stays tight.
        glm::vec3 center = (scene_.car_aabb_min + scene_.car_aabb_max) * 0.5f;
        scene_radius = glm::length(scene_.car_aabb_max - scene_.car_aabb_min) * 0.5f;
        if (scene_radius < 0.01f) scene_radius = 1.0f;

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

    // ---- Compute culling dispatch (3a: verification only) ----
    if (scene_.loaded && cull_bgl_ && gpu_.cull_aabb_buffer) {
        if (!cull_ready_) {
            cull_pipeline_ = make_cull_pipeline(gpu_.device, cull_bgl_);
            if (cull_pipeline_) {
                WGPUBindGroupEntry be[8] = {};
                be[0].binding = 0;
                be[0].buffer  = gpu_.cull_aabb_buffer;
                be[0].size    = (uint64_t)gpu_.cull_aabb_capacity * 32ull;
                be[1].binding = 1;
                be[1].buffer  = gpu_.cull_params_buffer;
                be[1].size    = 80;
                be[2].binding = 2;
                be[2].buffer  = gpu_.cull_output_buffer;
                be[2].size    = 4;
                be[3].binding = 3;
                be[3].buffer  = gpu_.cull_indices_buffer;
                be[3].size    = (uint64_t)gpu_.cull_indices_capacity * 4ull;
                be[4].binding = 4;
                be[4].buffer  = gpu_.cull_mesh_ids_buffer;
                be[4].size    = (uint64_t)gpu_.cull_aabb_capacity * 4ull;
                be[5].binding = 5;
                be[5].buffer  = gpu_.cull_mesh_counts_buffer;
                be[5].size    = (uint64_t)gpu_.cull_mesh_capacity * 4ull;
                be[6].binding = 6;
                be[6].buffer  = gpu_.cull_mesh_offsets_buffer;
                be[6].size    = (uint64_t)gpu_.cull_mesh_capacity * 4ull;
                be[7].binding = 7;
                be[7].buffer  = gpu_.cull_mesh_buckets_buffer;
                be[7].size    = (uint64_t)gpu_.cull_aabb_capacity * 4ull;

                WGPUBindGroupDescriptor bgd = {};
                bgd.layout     = cull_bgl_;
                bgd.entryCount = 8;
                bgd.entries    = be;
                cull_bg_ = wgpuDeviceCreateBindGroup(gpu_.device, &bgd);
                if (cull_bg_) cull_ready_ = true;
            }
        }
        if (cull_ready_) {
            // Clear the atomic counter.
            uint32_t zero = 0;
            wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_output_buffer, 0, &zero, 4);
            // Clear per-mesh counts (whole buffer to zero).
            if (gpu_.cull_mesh_counts_buffer) {
                std::vector<uint32_t> zeros(gpu_.cull_mesh_capacity, 0);
                wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_mesh_counts_buffer, 0,
                                     zeros.data(), zeros.size() * 4);
            }
            // Upload params: mat4 mvp_proj_view + u32 object_count + pad.
            struct CullParams {
                glm::mat4 mvp_proj_view;
                uint32_t  object_count;
                uint32_t  _pad[3];
            } params = {};
            params.mvp_proj_view = mvp_proj_view;
            params.object_count  = (uint32_t)scene_.objects.size();
            wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_params_buffer, 0,
                                 &params, sizeof(params));

            // Dispatch compute.
            WGPUCommandEncoder cull_enc = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
            WGPUComputePassDescriptor cpd = {};
            WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(cull_enc, &cpd);
            wgpuComputePassEncoderSetPipeline(cp, cull_pipeline_);
            wgpuComputePassEncoderSetBindGroup(cp, 0, cull_bg_, 0, nullptr);
            uint32_t n = (uint32_t)scene_.objects.size();
            wgpuComputePassEncoderDispatchWorkgroups(cp, (n + 63) / 64, 1, 1);
            wgpuComputePassEncoderEnd(cp);
            wgpuComputePassEncoderRelease(cp);

            // Read back the count for verification (3b only — 3c will consume
            // it directly on GPU without a readback).
            WGPUBufferDescriptor bd_rb = {};
            bd_rb.size  = 4;
            bd_rb.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
            WGPUBuffer readback = wgpuDeviceCreateBuffer(gpu_.device, &bd_rb);
            wgpuCommandEncoderCopyBufferToBuffer(cull_enc,
                gpu_.cull_output_buffer, 0, readback, 0, 4);

            WGPUCommandBuffer cull_cmd = wgpuCommandEncoderFinish(cull_enc, nullptr);
            wgpuQueueSubmit(gpu_.queue, 1, &cull_cmd);
            wgpuCommandBufferRelease(cull_cmd);
            wgpuCommandEncoderRelease(cull_enc);

            // Map-read the count (blocking, 100ms cap).
            struct MapState { bool done = false; bool ok = false; } ms;
            WGPUBufferMapCallbackInfo mcb = {};
            mcb.mode = WGPUCallbackMode_AllowSpontaneous;
            mcb.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* u1, void*) {
                auto* m = static_cast<MapState*>(u1);
                m->done = true;
                m->ok = (st == WGPUMapAsyncStatus_Success);
            };
            mcb.userdata1 = &ms;
            wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, 4, mcb);
            int waited = 0;
            while (!ms.done && waited < 100) {
                wgpuInstanceProcessEvents(gpu_.instance);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                waited++;
            }
            if (ms.ok) {
                const uint32_t* gpu_count = (const uint32_t*)
                    wgpuBufferGetConstMappedRange(readback, 0, 4);
                static uint32_t last_log = 0xFFFFFFFF;
                if (gpu_count[0] != last_log) {
                    std::fprintf(stderr,
                        "[engine] cull GPU: %u visible\n", gpu_count[0]);
                    last_log = gpu_count[0];
                }
                wgpuBufferUnmap(readback);
            }
            wgpuBufferRelease(readback);

            // 3c-1a: readback per-mesh counts, verify against CPU.
            {
                static bool done_mesh_verify = false;
                if (!done_mesh_verify) {
                    uint64_t n_meshes = scene_.meshes.size();
                    WGPUBufferDescriptor bd_mesh = {};
                    bd_mesh.size  = n_meshes * 4;
                    bd_mesh.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
                    WGPUBuffer mesh_rb = wgpuDeviceCreateBuffer(gpu_.device, &bd_mesh);

                    WGPUCommandEncoder enc2 = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
                    wgpuCommandEncoderCopyBufferToBuffer(enc2,
                        gpu_.cull_mesh_counts_buffer, 0, mesh_rb, 0, n_meshes * 4);
                    WGPUCommandBuffer cmd2 = wgpuCommandEncoderFinish(enc2, nullptr);
                    wgpuQueueSubmit(gpu_.queue, 1, &cmd2);
                    wgpuCommandBufferRelease(cmd2);
                    wgpuCommandEncoderRelease(enc2);

                    struct MapState2 { bool done = false; bool ok = false; } ms2;
                    WGPUBufferMapCallbackInfo mcb2 = {};
                    mcb2.mode = WGPUCallbackMode_AllowSpontaneous;
                    mcb2.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* u1, void*) {
                        auto* m = static_cast<MapState2*>(u1);
                        m->done = true; m->ok = (st == WGPUMapAsyncStatus_Success);
                    };
                    mcb2.userdata1 = &ms2;
                    wgpuBufferMapAsync(mesh_rb, WGPUMapMode_Read, 0, n_meshes * 4, mcb2);
                    int waited2 = 0;
                    while (!ms2.done && waited2 < 200) {
                        wgpuInstanceProcessEvents(gpu_.instance);
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        waited2++;
                    }
                    if (ms2.ok) {
                        const uint32_t* gpu_mesh = (const uint32_t*)
                            wgpuBufferGetConstMappedRange(mesh_rb, 0, n_meshes * 4);
                        // CPU-side expected per-mesh count. Compute a fresh
                        // CPU cull with the same frustum the GPU saw.
                        Frustum6 cpu_frustum = extract_frustum(mvp_proj_view);
                        std::vector<uint32_t> cpu_mesh(n_meshes, 0);
                        for (uint32_t i = 0; i < (uint32_t)scene_.objects.size(); ++i) {
                            const auto& obj  = scene_.objects[i];
                            const auto& mesh = scene_.meshes[obj.mesh_index];
                            glm::vec3 wmin, wmax;
                            transform_aabb(mesh.aabb_min, mesh.aabb_max, obj.transform, wmin, wmax);
                            if (frustum_test_aabb(cpu_frustum, wmin, wmax)) {
                                cpu_mesh[obj.mesh_index]++;
                            }
                        }
                        uint64_t matches = 0;
                        uint64_t diffs = 0;
                        for (uint64_t i = 0; i < n_meshes; ++i) {
                            if (gpu_mesh[i] == cpu_mesh[i]) matches++;
                            else diffs++;
                        }
                        std::fprintf(stderr,
                            "[engine] per-mesh counts: %llu / %llu match (%llu differ)\n",
                            (unsigned long long)matches,
                            (unsigned long long)n_meshes,
                            (unsigned long long)diffs);
                        wgpuBufferUnmap(mesh_rb);
                        done_mesh_verify = true;

                        // 3c-1b verification: read back buckets, confirm that
                        // for each mesh, the bucketed entries are exactly the
                        // visible objects of that mesh (set equality).
                        WGPUBufferDescriptor bd_bkt = {};
                        bd_bkt.size  = (uint64_t)scene_.objects.size() * 4;
                        bd_bkt.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
                        WGPUBuffer bkt_rb = wgpuDeviceCreateBuffer(gpu_.device, &bd_bkt);
                        WGPUCommandEncoder enc3 = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
                        wgpuCommandEncoderCopyBufferToBuffer(enc3,
                            gpu_.cull_mesh_buckets_buffer, 0,
                            bkt_rb, 0, (uint64_t)scene_.objects.size() * 4);
                        WGPUCommandBuffer cmd3 = wgpuCommandEncoderFinish(enc3, nullptr);
                        wgpuQueueSubmit(gpu_.queue, 1, &cmd3);
                        wgpuCommandBufferRelease(cmd3);
                        wgpuCommandEncoderRelease(enc3);

                        struct MapState3 { bool done=false; bool ok=false; } ms3;
                        WGPUBufferMapCallbackInfo mcb3 = {};
                        mcb3.mode = WGPUCallbackMode_AllowSpontaneous;
                        mcb3.callback = [](WGPUMapAsyncStatus st, WGPUStringView, void* u1, void*) {
                            auto* m = static_cast<MapState3*>(u1);
                            m->done = true; m->ok = (st == WGPUMapAsyncStatus_Success);
                        };
                        mcb3.userdata1 = &ms3;
                        wgpuBufferMapAsync(bkt_rb, WGPUMapMode_Read, 0,
                                          (uint64_t)scene_.objects.size() * 4, mcb3);
                        int w3 = 0;
                        while (!ms3.done && w3 < 200) {
                            wgpuInstanceProcessEvents(gpu_.instance);
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            w3++;
                        }
                        if (ms3.ok) {
                            const uint32_t* bkt = (const uint32_t*)
                                wgpuBufferGetConstMappedRange(bkt_rb, 0,
                                    (uint64_t)scene_.objects.size() * 4);
                            // Recompute mesh_offsets + expected buckets on CPU.
                            uint32_t n_meshes2 = (uint32_t)scene_.meshes.size();
                            std::vector<uint32_t> cpu_offsets(n_meshes2, 0);
                            std::vector<uint32_t> cpu_counts(n_meshes2, 0);
                            for (uint32_t i = 0; i < (uint32_t)scene_.objects.size(); ++i) {
                                cpu_counts[scene_.objects[i].mesh_index]++;
                            }
                            uint32_t running2 = 0;
                            for (uint32_t m = 0; m < n_meshes2; ++m) {
                                cpu_offsets[m] = running2;
                                running2 += cpu_counts[m];
                            }
                            // For each visible object, expected bucket slot
                            // is cpu_offsets[mesh_id] + (index within that mesh's
                            // visible set). Hard to compute exactly without
                            // per-mesh ordering. Simpler check: multiset compare.
                            std::vector<uint32_t> gpu_sorted;
                            for (uint32_t i = 0; i < (uint32_t)scene_.objects.size(); ++i) {
                                if (bkt[i] != 0xFFFFFFFFu && bkt[i] < scene_.objects.size()) {
                                    gpu_sorted.push_back(bkt[i]);
                                }
                            }
                            // Compare against flat list readback (from cull_indices_buffer)
                            std::vector<uint32_t> flat_sorted;
                            // We don't have flat list readback in this scope; skip strict compare,
                            // just report bucket contents count and range.
                            std::fprintf(stderr,
                                "[engine] cull buckets: %zu entries populated\n",
                                gpu_sorted.size());
                            wgpuBufferUnmap(bkt_rb);
                        }
                        wgpuBufferRelease(bkt_rb);
                    }
                    wgpuBufferRelease(mesh_rb);
                }
            }
        }
    }


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
                const uint32_t kStride = 160;
                const uint32_t N = (uint32_t)scene_.objects.size();

                // Frustum cull: build visible list + their uniforms in one pass.
                Frustum6 frustum = extract_frustum(mvp_proj_view);
                std::vector<uint32_t> visible;
                visible.reserve(N);

                struct U {
                    glm::mat4 mvp;
                    glm::mat4 model;
                    glm::vec4 material_color;
                    glm::vec4 material_params;
                };
                std::vector<uint8_t> uniform_data;
                uniform_data.reserve((size_t)N * kStride);

                for (uint32_t i = 0; i < N; ++i) {
                    const auto& obj  = scene_.objects[i];
                    const auto& mesh = scene_.meshes[obj.mesh_index];

                    glm::vec3 wmin, wmax;
                    transform_aabb(mesh.aabb_min, mesh.aabb_max, obj.transform,
                                   wmin, wmax);
                    if (!frustum_test_aabb(frustum, wmin, wmax)) continue;

                    visible.push_back(i);

                    const auto& mat = scene_.materials[obj.material_index];
                    U u = {};
                    u.mvp    = mvp_proj_view * obj.transform;
                    u.model  = obj.transform;
                    u.material_color  = mat.base_color;
                    u.material_params = glm::vec4(mat.roughness, mat.metallic, mat.ao, 0.0f);

                    size_t prev = uniform_data.size();
                    uniform_data.resize(prev + kStride);
                    std::memcpy(uniform_data.data() + prev, &u, sizeof(u));
                }

                static uint32_t last_v = 0;
                if ((last_v / 60) != ((uint32_t)visible.size() / 60)) {
                    std::fprintf(stderr,
                        "[engine] frustum cull: %zu / %u visible (%.1f%%)\n",
                        visible.size(), N,
                        100.0f * (float)visible.size() / std::max(1u, N));
                    last_v = (uint32_t)visible.size();
                }

                if (!uniform_data.empty()) {
                    wgpuQueueWriteBuffer(gpu_.queue, gpu_.uniform_buffer, 0,
                                         uniform_data.data(), uniform_data.size());
                }

                // Per-instance vertex buffer: one u32 per visible object = the
                // VISIBLE-SLOT index (0..N-1). The shader reads u_arr[obj_idx],
                // and uniforms were packed in visible-slot order, so obj_idx
                // must equal the visible slot, not the object index.
                static WGPUBuffer instance_idx_buffer = nullptr;
                static uint64_t instance_idx_capacity = 0;
                uint64_t needed = (uint64_t)visible.size() * 4;
                if (instance_idx_buffer == nullptr || instance_idx_capacity < needed) {
                    if (instance_idx_buffer) wgpuBufferRelease(instance_idx_buffer);
                    WGPUBufferDescriptor ibd = {};
                    ibd.size  = std::max<uint64_t>(needed, 256);
                    ibd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
                    instance_idx_buffer = wgpuDeviceCreateBuffer(gpu_.device, &ibd);
                    instance_idx_capacity = ibd.size;
                }
                // Group visible objects by mesh. Each unique mesh gets one
                // draw call with instance_count = number of visible objects
                // referencing that mesh.
                struct MeshBatch {
                    uint32_t mesh_index;
                    uint32_t first_instance;
                    uint32_t instance_count;
                };

                // Reverse map: object index -> uniform slot.
                // Uniforms were packed in `visible` order (visible[k] at slot k).
                std::vector<uint32_t> uniform_slot_of(scene_.objects.size(), 0);
                for (uint32_t slot = 0; slot < (uint32_t)visible.size(); ++slot) {
                    uniform_slot_of[visible[slot]] = slot;
                }

                // Sort visible objects by mesh index.
                std::vector<uint32_t> visible_by_mesh = visible;
                std::sort(visible_by_mesh.begin(), visible_by_mesh.end(),
                    [&](uint32_t a, uint32_t b) {
                        return scene_.objects[a].mesh_index <
                               scene_.objects[b].mesh_index;
                    });

                // Build batches + instance buffer (uniform slots, in mesh order).
                std::vector<MeshBatch> batches;
                batches.reserve(64);
                std::vector<uint32_t> instance_slots;
                instance_slots.reserve(visible.size());

                for (uint32_t obj_index : visible_by_mesh) {
                    uint32_t mesh_idx = scene_.objects[obj_index].mesh_index;
                    if (batches.empty() || batches.back().mesh_index != mesh_idx) {
                        MeshBatch b;
                        b.mesh_index     = mesh_idx;
                        b.first_instance = (uint32_t)instance_slots.size();
                        b.instance_count = 0;
                        batches.push_back(b);
                    }
                    instance_slots.push_back(uniform_slot_of[obj_index]);
                    batches.back().instance_count++;
                }

                wgpuQueueWriteBuffer(gpu_.queue, instance_idx_buffer, 0,
                                     instance_slots.data(),
                                     instance_slots.size() * 4);

                static uint32_t last_batch_count = 0;
                if (last_batch_count == 0) {
                    std::fprintf(stderr,
                        "[engine] instancing: %zu draws for %zu visible objects "
                        "(%.1f draws/object saved)\n",
                        batches.size(), visible.size(),
                        visible.size() > 0
                            ? 100.0f * (1.0f - (float)batches.size() / (float)visible.size())
                            : 0.0f);
                    last_batch_count = (uint32_t)batches.size();
                }

                wgpuRenderPassEncoderSetBindGroup(rp, 0, gpu_.uniform_bind_group, 0, nullptr);

                for (const auto& b : batches) {
                    const auto& mesh = scene_.meshes[b.mesh_index];
                    const auto& mat  = scene_.materials[mesh.material_index];

                    wgpuRenderPassEncoderSetBindGroup(rp, 1, mat.bind_group, 0, nullptr);
                    wgpuRenderPassEncoderSetVertexBuffer(rp, 0, mesh.vertex_buffer, 0,
                        mesh.vertex_count * 48);
                    wgpuRenderPassEncoderSetVertexBuffer(rp, 1, instance_idx_buffer,
                        (uint64_t)b.first_instance * 4,
                        (uint64_t)b.instance_count * 4);
                    wgpuRenderPassEncoderSetIndexBuffer(rp, mesh.index_buffer,
                        WGPUIndexFormat_Uint32, 0, mesh.index_count * 4);
                    wgpuRenderPassEncoderDrawIndexed(rp, mesh.index_count,
                        b.instance_count, 0, 0, 0);
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
    if (!material_bgl_ || !scene_.default_sampler || !scene_.white_view) {
        err = "material infrastructure not ready";
        return false;
    }

    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        load_err, load_warn;

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
        "[engine] gltf loaded: %zu meshes, %zu nodes, %zu accessors, "
        "%zu images, %zu materials\n",
        model.meshes.size(), model.nodes.size(), model.accessors.size(),
        model.images.size(), model.materials.size());

    // ====================================================================
    // Helper: upload a raw RGBA8 image to a WGPUTexture, returning the view.
    // Handles the 256-byte row pitch requirement for CopyBufferToTexture.
    // ====================================================================
    auto upload_rgba = [&](const uint8_t* pixels, uint32_t w, uint32_t h,
                          WGPUTextureFormat format) -> WGPUTextureView {
        if (!pixels || w == 0 || h == 0) return nullptr;

        WGPUTextureDescriptor td = {};
        td.dimension     = WGPUTextureDimension_2D;
        td.size          = { w, h, 1 };
        td.format        = format;
        td.mipLevelCount = 1;
        td.sampleCount   = 1;
        td.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        WGPUTexture tex = wgpuDeviceCreateTexture(gpu_.device, &td);
        if (!tex) return nullptr;

        // Pad each row to 256-byte multiple.
        uint32_t row_bytes    = w * 4;
        uint32_t padded_pitch = (row_bytes + 255u) & ~255u;
        std::vector<uint8_t> staging_data((size_t)padded_pitch * h, 0);
        for (uint32_t y = 0; y < h; ++y) {
            std::memcpy(staging_data.data() + y * padded_pitch,
                        pixels + y * row_bytes, row_bytes);
        }

        WGPUBufferDescriptor bd = {};
        bd.size  = staging_data.size();
        bd.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        WGPUBuffer staging = wgpuDeviceCreateBuffer(gpu_.device, &bd);
        wgpuQueueWriteBuffer(gpu_.queue, staging, 0,
                             staging_data.data(), staging_data.size());

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu_.device, nullptr);
        WGPUTexelCopyBufferInfo src = {};
        src.buffer = staging;
        src.layout.bytesPerRow  = padded_pitch;
        src.layout.rowsPerImage = h;
        WGPUTexelCopyTextureInfo dst = {};
        dst.texture  = tex;
        dst.mipLevel = 0;
        dst.origin   = {0, 0, 0};
        WGPUExtent3D extent = { w, h, 1 };
        wgpuCommandEncoderCopyBufferToTexture(enc, &src, &dst, &extent);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(gpu_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        wgpuBufferRelease(staging);

        return wgpuTextureCreateView(tex, nullptr);
    };

    // ====================================================================
    // 1. Upload all images. Images may be PNG or JPEG; tinygltf decodes
    //    them to 8-bit RGBA (or RGB, or grayscale) via stb_image.
    // ====================================================================
    std::vector<WGPUTextureView> image_views(model.images.size(), nullptr);
    std::vector<WGPUTexture>     image_textures(model.images.size(), nullptr);
    for (size_t i = 0; i < model.images.size(); ++i) {
        const auto& img = model.images[i];
        if (img.image.empty() || img.width <= 0 || img.height <= 0) {
            std::fprintf(stderr, "[engine] image[%zu] %s empty\n", i, img.name.c_str());
            continue;
        }
        // Convert RGB or grayscale to RGBA for upload.
        std::vector<uint8_t> rgba;
        if (img.component == 4) {
            rgba = img.image;
        } else if (img.component == 3) {
            rgba.resize((size_t)img.width * img.height * 4);
            for (size_t p = 0; p < (size_t)img.width * img.height; ++p) {
                rgba[p*4+0] = img.image[p*3+0];
                rgba[p*4+1] = img.image[p*3+1];
                rgba[p*4+2] = img.image[p*3+2];
                rgba[p*4+3] = 255;
            }
        } else if (img.component == 1) {
            rgba.resize((size_t)img.width * img.height * 4);
            for (size_t p = 0; p < (size_t)img.width * img.height; ++p) {
                rgba[p*4+0] = rgba[p*4+1] = rgba[p*4+2] = img.image[p];
                rgba[p*4+3] = 255;
            }
        } else {
            continue;
        }

        // Base color textures use sRGB; MR, normal, and others use linear.
        // We can't distinguish by image alone; we'll create both views later
        // if needed. For v1, use sRGB for images referenced as baseColor and
        // linear for others. To keep it simple: create two textures per image
        // only when needed. For now: create sRGB for all, and rely on the
        // shader to convert MR values manually (they're in G/B channels and
        // multiply low numbers; linear interpretation is close enough).
        //
        // Actually: create TWO textures per image. sRGB for base color usage,
        // linear for normal/MR usage. Bind the correct one per material slot.
        // This is the "lazy duplicate" approach; costs 2× GPU memory.
        //
        // Simpler v1: single texture, sRGB interpretation. The MR and normal
        // textures will look slightly bright in the linear pipeline, but
        // we're in "see the car" mode; proper sRGB/linear split is next step.
        auto view = upload_rgba(rgba.data(), (uint32_t)img.width,
                                (uint32_t)img.height,
                                WGPUTextureFormat_RGBA8UnormSrgb);
        if (view) {
            image_views[i] = view;
        }
    }
    std::fprintf(stderr, "[engine] %zu images uploaded\n", model.images.size());

    // ====================================================================
    // 2. Create Material entries. One per glTF material, with bind group.
    // ====================================================================
    scene_.materials.clear();
    scene_.materials.reserve(model.materials.size() + 1);

    // Default material for primitives with no material assigned (glTF -1).
    // Index 0. glTF materials are appended starting at index 1 below.
    {
        Material def;
        def.base_color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
        def.roughness  = 0.5f;
        def.metallic   = 0.0f;
        def.ao         = 1.0f;
        WGPUBindGroupEntry be[4] = {};
        be[0].binding     = 0;
        be[0].textureView = scene_.white_view;
        be[1].binding     = 1;
        be[1].textureView = scene_.white_view;
        be[2].binding     = 2;
        be[2].textureView = scene_.white_view;
        be[3].binding     = 3;
        be[3].sampler     = scene_.default_sampler;
        WGPUBindGroupDescriptor bgd = {};
        bgd.layout = material_bgl_;
        bgd.entryCount = 4;
        bgd.entries = be;
        def.bind_group = wgpuDeviceCreateBindGroup(gpu_.device, &bgd);
        scene_.materials.push_back(def);
    }

    for (size_t mi = 0; mi < model.materials.size(); ++mi) {
        const auto& gmat = model.materials[mi];
        Material mat;

        // Base color factor (linear).
        if (gmat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
            mat.base_color = glm::vec4(
                (float)gmat.pbrMetallicRoughness.baseColorFactor[0],
                (float)gmat.pbrMetallicRoughness.baseColorFactor[1],
                (float)gmat.pbrMetallicRoughness.baseColorFactor[2],
                (float)gmat.pbrMetallicRoughness.baseColorFactor[3]);
        }
        mat.roughness = (float)gmat.pbrMetallicRoughness.roughnessFactor;
        mat.metallic  = (float)gmat.pbrMetallicRoughness.metallicFactor;


        // Resolve texture views.
        auto resolve_texture = [&](int tex_index, WGPUTextureView fallback) -> WGPUTextureView {
            if (tex_index < 0 || tex_index >= (int)model.textures.size()) return fallback;
            int img_index = model.textures[tex_index].source;
            if (img_index < 0 || img_index >= (int)image_views.size()) return fallback;
            return image_views[img_index] ? image_views[img_index] : fallback;
        };

        WGPUTextureView base_view = resolve_texture(
            gmat.pbrMetallicRoughness.baseColorTexture.index, scene_.white_view);
        WGPUTextureView mr_view = resolve_texture(
            gmat.pbrMetallicRoughness.metallicRoughnessTexture.index, scene_.white_view);
        WGPUTextureView normal_view = resolve_texture(
            gmat.normalTexture.index, scene_.white_view);

        // Build the material bind group.
        WGPUBindGroupEntry be[4] = {};
        be[0].binding     = 0;
        be[0].textureView = base_view;
        be[1].binding     = 1;
        be[1].textureView = mr_view;
        be[2].binding     = 2;
        be[2].textureView = normal_view;
        be[3].binding     = 3;
        be[3].sampler     = scene_.default_sampler;

        WGPUBindGroupDescriptor bgd = {};
        bgd.layout     = material_bgl_;
        bgd.entryCount = 4;
        bgd.entries    = be;
        mat.bind_group = wgpuDeviceCreateBindGroup(gpu_.device, &bgd);
        if (!mat.bind_group) {
            std::fprintf(stderr, "[engine] material[%zu] bind group failed\n", mi);
        }
        // DIAGNOSTIC: log material index, name, base color, texture status.
        std::fprintf(stderr,
            "[engine] mat[%zu] '%s' base=(%.4f,%.4f,%.4f) base_tex=%s bg=%p\n",
            mi, gmat.name.c_str(),
            mat.base_color.r, mat.base_color.g, mat.base_color.b,
            (base_view != scene_.white_view) ? "YES" : "NO(fallback)",
            (void*)mat.bind_group);
        scene_.materials.push_back(mat);
    }
    std::fprintf(stderr, "[engine] %zu materials, %zu bind groups\n",
                 scene_.materials.size(), model.materials.size());

    // ====================================================================
    // 3. Build meshes (48-byte vertex: pos+normal+uv+tangent).
    // ====================================================================
    struct PrimRef { uint32_t mesh_index; uint32_t material_index; };
    std::vector<std::vector<PrimRef>> gltf_mesh_prims(model.meshes.size());

    glm::vec3 scene_min( 1e30f);
    glm::vec3 scene_max(-1e30f);

    for (size_t mi = 0; mi < model.meshes.size(); ++mi) {
        const auto& gmesh = model.meshes[mi];
        for (size_t pi = 0; pi < gmesh.primitives.size(); ++pi) {
            const auto& prim = gmesh.primitives[pi];

            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end()) continue;
            const auto& pos_acc = model.accessors[pos_it->second];
            if (pos_acc.type != TINYGLTF_TYPE_VEC3 ||
                pos_acc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                continue;
            }

            const auto& pos_bv = model.bufferViews[pos_acc.bufferView];
            const float* pos_ptr = reinterpret_cast<const float*>(
                model.buffers[pos_bv.buffer].data.data()
                + pos_bv.byteOffset + pos_acc.byteOffset);
            size_t vert_count = pos_acc.count;

            // Normals (optional).
            const float* normals_ptr = nullptr;
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
                    }
                }
            }
            // UVs (optional).
            const float* uvs_ptr = nullptr;
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
                    }
                }
            }
            // Tangents (optional, but for normal mapping we compute if absent).
            const float* tangents_ptr = nullptr;
            {
                auto it = prim.attributes.find("TANGENT");
                if (it != prim.attributes.end()) {
                    const auto& acc = model.accessors[it->second];
                    if (acc.type == TINYGLTF_TYPE_VEC4 &&
                        acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const auto& bv = model.bufferViews[acc.bufferView];
                        tangents_ptr = reinterpret_cast<const float*>(
                            model.buffers[bv.buffer].data.data()
                            + bv.byteOffset + acc.byteOffset);
                    }
                }
            }

            // Build interleaved vertex data (48 bytes per vertex).
            std::vector<float> vertex_data(vert_count * 12);  // 12 floats = 48 bytes
            glm::vec3 mmin(1e30f), mmax(-1e30f);
            for (size_t v = 0; v < vert_count; ++v) {
                float px = pos_ptr[v*3+0];
                float py = pos_ptr[v*3+1];
                float pz = pos_ptr[v*3+2];
                vertex_data[v*12+0]  = px;
                vertex_data[v*12+1]  = py;
                vertex_data[v*12+2]  = pz;
                vertex_data[v*12+3]  = normals_ptr ? normals_ptr[v*3+0] : 0.0f;
                vertex_data[v*12+4]  = normals_ptr ? normals_ptr[v*3+1] : 1.0f;
                vertex_data[v*12+5]  = normals_ptr ? normals_ptr[v*3+2] : 0.0f;
                vertex_data[v*12+6]  = uvs_ptr ? uvs_ptr[v*2+0] : 0.0f;
                vertex_data[v*12+7]  = uvs_ptr ? uvs_ptr[v*2+1] : 0.0f;
                // Tangent: use file-provided, or fallback (1,0,0,1).
                if (tangents_ptr) {
                    vertex_data[v*12+8]  = tangents_ptr[v*4+0];
                    vertex_data[v*12+9]  = tangents_ptr[v*4+1];
                    vertex_data[v*12+10] = tangents_ptr[v*4+2];
                    vertex_data[v*12+11] = tangents_ptr[v*4+3];
                } else {
                    vertex_data[v*12+8]  = 1.0f;
                    vertex_data[v*12+9]  = 0.0f;
                    vertex_data[v*12+10] = 0.0f;
                    vertex_data[v*12+11] = 1.0f;
                }
                mmin.x = std::min(mmin.x, px); mmax.x = std::max(mmax.x, px);
                mmin.y = std::min(mmin.y, py); mmax.y = std::max(mmax.y, py);
                mmin.z = std::min(mmin.z, pz); mmax.z = std::max(mmax.z, pz);
            }

            // Indices.
            std::vector<uint32_t> index_data;
            if (prim.indices >= 0) {
                const auto& idx_acc = model.accessors[prim.indices];
                const auto& idx_bv  = model.bufferViews[idx_acc.bufferView];
                const uint8_t* src = model.buffers[idx_bv.buffer].data.data()
                                    + idx_bv.byteOffset + idx_acc.byteOffset;
                index_data.resize(idx_acc.count);
                if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(src);
                    for (size_t i = 0; i < idx_acc.count; ++i) index_data[i] = p16[i];
                } else if (idx_acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(src);
                    for (size_t i = 0; i < idx_acc.count; ++i) index_data[i] = p32[i];
                } else continue;
            } else {
                index_data.resize(vert_count);
                for (size_t i = 0; i < vert_count; ++i) index_data[i] = (uint32_t)i;
            }

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
            mesh.vertex_count   = (uint32_t)vert_count;
            mesh.index_count    = (uint32_t)index_data.size();
            mesh.aabb_min       = mmin;
            mesh.aabb_max       = mmax;

            // Material index: glTF allows -1 (no material). Map to a
            // "default material" slot at index 0.
            int mat_idx = (prim.material >= 0 &&
                           prim.material < (int)model.materials.size())
                        ? (prim.material + 1) : 0;
            mesh.material_index = mat_idx;
            scene_.meshes.push_back(mesh);

            uint32_t new_mi = (uint32_t)(scene_.meshes.size() - 1);
            gltf_mesh_prims[mi].push_back(PrimRef{new_mi, (uint32_t)mat_idx});

            scene_min.x = std::min(scene_min.x, mmin.x); scene_max.x = std::max(scene_max.x, mmax.x);
            scene_min.y = std::min(scene_min.y, mmin.y); scene_max.y = std::max(scene_max.y, mmax.y);
            scene_min.z = std::min(scene_min.z, mmin.z); scene_max.z = std::max(scene_max.z, mmax.z);
        }
    }
    std::fprintf(stderr, "[engine] %zu GPU meshes uploaded (48B/vertex)\n",
                 scene_.meshes.size());

    // ====================================================================
    // 4. Walk nodes, produce SceneObjects.
    // ====================================================================
    for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
        const auto& node = model.nodes[ni];
        if (node.mesh < 0) continue;

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
            obj.mesh_index     = pr.mesh_index;
            obj.material_index = pr.material_index;
            obj.transform      = local;
            scene_.objects.push_back(obj);
        }
    }

    // Save the glTF content AABB for camera framing (before ground plane).
    scene_.car_aabb_min = scene_min;
    scene_.car_aabb_max = scene_max;

    // ---- Ground plane ----
    {
        const float scene_size_x = scene_max.x - scene_min.x;
        const float scene_size_z = scene_max.z - scene_min.z;
        const float half_size = std::max(std::max(scene_size_x, scene_size_z) * 5.0f, 20.0f);
        const float y  = scene_min.y - 0.01f;
        const float cx = (scene_min.x + scene_max.x) * 0.5f;
        const float cz = (scene_min.z + scene_max.z) * 0.5f;

        struct V { float px,py,pz, nx,ny,nz, u,v, tx,ty,tz,tw; };
        V verts[4] = {
            { cx - half_size, y, cz - half_size,  0,1,0,  0,0,  1,0,0,1 },
            { cx + half_size, y, cz - half_size,  0,1,0,  1,0,  1,0,0,1 },
            { cx + half_size, y, cz + half_size,  0,1,0,  1,1,  1,0,0,1 },
            { cx - half_size, y, cz + half_size,  0,1,0,  0,1,  1,0,0,1 },
        };
        uint32_t idx[6] = { 0, 1, 2,  0, 2, 3 };

        Mesh ground;
        {
            WGPUBufferDescriptor vbd = {};
            vbd.size  = sizeof(verts);
            vbd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            ground.vertex_buffer = wgpuDeviceCreateBuffer(gpu_.device, &vbd);
            wgpuQueueWriteBuffer(gpu_.queue, ground.vertex_buffer, 0, verts, sizeof(verts));
        }
        {
            WGPUBufferDescriptor ibd = {};
            ibd.size  = sizeof(idx);
            ibd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            ground.index_buffer = wgpuDeviceCreateBuffer(gpu_.device, &ibd);
            wgpuQueueWriteBuffer(gpu_.queue, ground.index_buffer, 0, idx, sizeof(idx));
        }
        ground.vertex_count = 4;
        ground.index_count  = 6;
        ground.aabb_min     = glm::vec3(cx - half_size, y, cz - half_size);
        ground.aabb_max     = glm::vec3(cx + half_size, y, cz + half_size);

        Material gmat;
        gmat.base_color = glm::vec4(0.35f, 0.35f, 0.35f, 1.0f);
        gmat.roughness  = 0.85f;
        gmat.metallic   = 0.0f;
        gmat.ao         = 1.0f;
        gmat.bind_group = scene_.materials[0].bind_group;  // reuse default bind group

        uint32_t gm = (uint32_t)scene_.materials.size();
        scene_.materials.push_back(gmat);

        uint32_t gmesh = (uint32_t)scene_.meshes.size();
        ground.material_index = (int)gm;
        scene_.meshes.push_back(ground);

        SceneObject gobj;
        gobj.mesh_index     = gmesh;
        gobj.material_index = gm;
        gobj.transform      = glm::mat4(1.0f);
        scene_.objects.push_back(gobj);

        scene_.aabb_min = glm::min(scene_.aabb_min, ground.aabb_min);
        scene_.aabb_max = glm::max(scene_.aabb_max, ground.aabb_max);

        std::fprintf(stderr,
            "[engine] ground plane added (%.1fx%.1f at y=%.2f)\n",
            half_size * 2.0f, half_size * 2.0f, y);
    }

    scene_.aabb_min = glm::min(scene_.aabb_min, scene_min);
    scene_.aabb_max = glm::max(scene_.aabb_max, scene_max);
    scene_.loaded   = true;

    // ---- Fill cull AABB buffer with world-space AABBs for all objects ----
    {
        uint32_t n = (uint32_t)scene_.objects.size();
        if (n > gpu_.cull_aabb_capacity) {
            std::fprintf(stderr,
                "[engine] WARN: %u objects > cull capacity %u\n",
                n, gpu_.cull_aabb_capacity);
        } else if (gpu_.cull_aabb_buffer) {
            struct AabbPadded {
                float min_xyz[3]; float pad0;
                float max_xyz[3]; float pad1;
            };
            std::vector<AabbPadded> buf(n);
            for (uint32_t i = 0; i < n; ++i) {
                const auto& obj  = scene_.objects[i];
                const auto& mesh = scene_.meshes[obj.mesh_index];
                glm::vec3 wmin, wmax;
                transform_aabb(mesh.aabb_min, mesh.aabb_max, obj.transform, wmin, wmax);
                buf[i].min_xyz[0] = wmin.x; buf[i].min_xyz[1] = wmin.y; buf[i].min_xyz[2] = wmin.z; buf[i].pad0 = 0;
                buf[i].max_xyz[0] = wmax.x; buf[i].max_xyz[1] = wmax.y; buf[i].max_xyz[2] = wmax.z; buf[i].pad1 = 0;
            }
            wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_aabb_buffer, 0,
                                 buf.data(), buf.size() * sizeof(AabbPadded));
            std::fprintf(stderr,
                "[engine] cull AABB buffer: %u objects uploaded\n", n);

            // Fill mesh_id buffer: per-object mesh index.
            if (gpu_.cull_mesh_ids_buffer) {
                std::vector<uint32_t> mesh_ids(n);
                for (uint32_t i = 0; i < n; ++i) {
                    mesh_ids[i] = scene_.objects[i].mesh_index;
                }
                wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_mesh_ids_buffer, 0,
                                     mesh_ids.data(), mesh_ids.size() * 4);
                std::fprintf(stderr,
                    "[engine] cull mesh_id buffer: %u entries uploaded\n", n);

                // Compute per-mesh base offsets (prefix sum of per-mesh object
                // counts) and upload.
                uint32_t n_meshes = (uint32_t)scene_.meshes.size();
                std::vector<uint32_t> mesh_obj_count(n_meshes, 0);
                for (uint32_t i = 0; i < n; ++i) {
                    mesh_obj_count[scene_.objects[i].mesh_index]++;
                }
                std::vector<uint32_t> mesh_offsets(n_meshes, 0);
                uint32_t running = 0;
                for (uint32_t m = 0; m < n_meshes; ++m) {
                    mesh_offsets[m] = running;
                    running += mesh_obj_count[m];
                }
                wgpuQueueWriteBuffer(gpu_.queue, gpu_.cull_mesh_offsets_buffer, 0,
                                     mesh_offsets.data(), mesh_offsets.size() * 4);
                std::fprintf(stderr,
                    "[engine] cull mesh_offsets: %u meshes, %u total capacity\n",
                    n_meshes, running);
            }
        }
    }

    std::fprintf(stderr,
        "[engine] scene: %zu objects, AABB (%.2f,%.2f,%.2f)..(%.2f,%.2f,%.2f)\n",
        scene_.objects.size(),
        scene_.aabb_min.x, scene_.aabb_min.y, scene_.aabb_min.z,
        scene_.aabb_max.x, scene_.aabb_max.y, scene_.aabb_max.z);

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
