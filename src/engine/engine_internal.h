#ifndef ENGINE_INTERNAL_H
#define ENGINE_INTERNAL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>
#include "frame_graph.h"

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

    // HZB (Hi-Z) depth pyramid. Mip 0 = converted depth buffer (R32Float),
    // mips 1..N = hierarchically downsampled minimum. Used by SSR/SSAO/SSGI.
    WGPUTexture     hzb_texture = nullptr;
    std::vector<WGPUTextureView> hzb_mip_views;      // all mips, for sampling
    std::vector<WGPUTextureView> hzb_storage_views;  // per-mip, for storage writes
    uint32_t        hzb_mip_count = 0;
    uint32_t        hzb_width = 0;
    uint32_t        hzb_height = 0;

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

// A mesh owns its GPU buffers. One glTF primitive = one Mesh.
struct Mesh {
    WGPUBuffer vertex_buffer = nullptr;
    WGPUBuffer index_buffer  = nullptr;
    uint32_t   vertex_count  = 0;
    uint32_t   index_count   = 0;
    glm::vec3  aabb_min      = glm::vec3(0.0f);
    glm::vec3  aabb_max      = glm::vec3(0.0f);

    // Material index into Scene::materials.
    int material_index = -1;
};

// A material references one texture per map (base, MR, normal), plus a
// pre-built bind group that binds all three textures + sampler at group 1.
// Materials without a given map bind a 1×1 white texture so the shader
// always has something valid to sample.
struct Material {
    glm::vec4 base_color = glm::vec4(1.0f);   // linear RGBA, multiplied by texture
    float     roughness  = 1.0f;
    float     metallic   = 0.0f;
    float     ao         = 1.0f;

    WGPUBindGroup bind_group = nullptr;        // group 1
};

// A scene object references a mesh and carries its world transform.
// Multiple objects can reference the same mesh (instancing at the object level,
// not yet at the draw level).
struct SceneObject {
    uint32_t  mesh_index     = 0;
    uint32_t  material_index = 0;
    glm::mat4 transform      = glm::mat4(1.0f);
};

struct Scene {
    std::vector<Mesh>        meshes;
    std::vector<Material>    materials;
    std::vector<SceneObject> objects;
    glm::vec3                aabb_min = glm::vec3(0.0f);      // full scene (incl. ground)
    glm::vec3                aabb_max = glm::vec3(0.0f);
    glm::vec3                car_aabb_min = glm::vec3(0.0f);  // glTF content only
    glm::vec3                car_aabb_max = glm::vec3(0.0f);
    bool                     loaded   = false;

    // GpuContext-owned resources for textures. Kept here for lifetime
    // management with the scene.
    WGPUSampler              default_sampler = nullptr;   // linear/linear/repeat
    WGPUTexture              white_tex = nullptr;         // 1×1 white
    WGPUTextureView          white_view = nullptr;
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
    FrameGraph fg_;
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

    // Material binding layout (group 1): base color + MR + normal + sampler.
    WGPUBindGroupLayout material_bgl_ = nullptr;

    // HZB compute pipelines and bind groups.
    WGPUComputePipeline hzb_convert_pipeline_  = nullptr;
    WGPUComputePipeline hzb_downsample_pipeline_ = nullptr;
    bool                hzb_pipelines_ready_   = false;
    WGPUBindGroupLayout hzb_convert_bgl_       = nullptr;
    WGPUBindGroupLayout hzb_downsample_bgl_    = nullptr;
    WGPUBindGroup       hzb_convert_bg_        = nullptr;  // depth -> mip0
    std::vector<WGPUBindGroup> hzb_downsample_bgs_;        // mipN-1 -> mipN

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
