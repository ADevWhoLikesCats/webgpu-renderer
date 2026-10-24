#ifndef ENGINE_INTERNAL_H
#define ENGINE_INTERNAL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>

#include <glm/glm.hpp>

struct GLFWwindow; /* unused; placeholder for future hosts */
struct btDiscreteDynamicsWorld;

namespace engine {

struct Surface {
    WGPUSurface surface = nullptr;
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    uint32_t width  = 0;
    uint32_t height = 0;
};

struct GpuContext {
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;

    // Depth target (attached to every offscreen render pass).
    WGPUTexture     depth_texture = nullptr;
    WGPUTextureView depth_view    = nullptr;

    // G-buffer (deferred geometry pass writes here, composite pass reads).
    WGPUTexture     gbuf_albedo_tex   = nullptr;
    WGPUTextureView gbuf_albedo_view  = nullptr;
    WGPUTexture     gbuf_normal_tex   = nullptr;
    WGPUTextureView gbuf_normal_view  = nullptr;
    WGPUTexture     gbuf_material_tex = nullptr;
    WGPUTextureView gbuf_material_view= nullptr;

    // Sampler for the composite pass to sample G-buffer textures.
    WGPUSampler gbuf_sampler = nullptr;

    // Per-frame uniforms (MVP + model matrix).
    WGPUBuffer        uniform_buffer    = nullptr;
    WGPUBindGroupLayout uniform_bgl     = nullptr;
    WGPUBindGroup     uniform_bind_group = nullptr;

    // Geometry.
    WGPUBuffer vertex_buffer = nullptr;
    uint32_t   vertex_count  = 0;
    uint32_t   vertex_stride = 32;  // pos(12) + normal(12) + uv(8)
    WGPUBuffer index_buffer  = nullptr;
    uint32_t   index_count   = 0;

    void destroy();
};

struct SceneObject {
    std::string name;
    glm::mat4   transform = glm::mat4(1.0f);
    // Future: mesh handle, material handle, physics body handle.
};

struct Scene {
    std::vector<SceneObject> objects;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool init(const char* app_name,
              int enable_validation,
              int prefer_software_device,
              void* wayland_surface,
              void* wayland_display,
              uint32_t width,
              uint32_t height,
              std::string& err);

    void shutdown();

    void resize(uint32_t width, uint32_t height);
    void begin_frame();
    void end_frame();
    void tick();

    bool load_gltf(const char* utf8_path, std::string& err);

    // Accessors for internal use.
    GpuContext& gpu() { return gpu_; }
    Surface&    surface() { return surface_; }
    Scene&      scene() { return scene_; }

    // Adapter info.
    void get_adapter_info(char* device, size_t device_cap,
                          char* desc,   size_t desc_cap,
                          uint32_t* backend, uint32_t* vendor, uint32_t* device_id);

private:
    GpuContext gpu_;
    Surface    surface_;
    Scene      scene_;
    bool       initialized_ = false;

    // Minimal triangle pipeline (checkpoint: first pixel).
    WGPURenderPipeline triangle_pipeline_ = nullptr;
    bool               triangle_pipeline_ready_ = false;

    // Geometry pipeline (writes G-buffer).
    WGPURenderPipeline geometry_pipeline_ = nullptr;
    bool               geometry_pipeline_ready_ = false;

    // Composite pipeline (reads G-buffer, writes final color).
    WGPURenderPipeline  composite_pipeline_ = nullptr;
    bool                composite_pipeline_ready_ = false;
    WGPUBindGroupLayout composite_bgl_ = nullptr;
    WGPUBindGroup       composite_bg_  = nullptr;

    // Offscreen render target (Path B: no swapchain in the loop).
    WGPUTexture offscreen_color_  = nullptr;
    WGPUTextureView offscreen_view_ = nullptr;
    WGPUTextureFormat offscreen_format_ = WGPUTextureFormat_BGRA8UnormSrgb;
    uint32_t offscreen_width_  = 0;
    uint32_t offscreen_height_ = 0;
    uint32_t frame_index_ = 0;
};

} // namespace engine

#endif
