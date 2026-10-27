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

    // Scene color (composite output, before SSR). Same format as offscreen_color_.
    // SSR reads this, modifies, and writes to offscreen_color_.
    WGPUTexture     scene_color_tex  = nullptr;
    WGPUTextureView scene_color_view = nullptr;

    // CSM (Cascaded Shadow Maps). 4 cascades, 1024x1024 each, Depth32Float.
    // shadow_view is a 2D array view (4 layers), used for sampling in composite.
    WGPUTexture      shadow_map   = nullptr;
    WGPUTextureView  shadow_view  = nullptr;
    WGPUSampler      shadow_sampler = nullptr;

    // Shadow-pass uniform buffer: one mat4 (mvp) per object, overwritten
    // per cascade (4 uploads per frame, one per cascade).
    WGPUBuffer       shadow_uniform_buffer = nullptr;
    uint32_t         shadow_uniform_capacity = 0;  // in objects

    // HZB (Hi-Z) depth pyramid. Mip 0 = converted depth buffer (R32Float),
    // mips 1..N = hierarchically downsampled minimum. Used by SSR/SSAO/SSGI.
    WGPUTexture     hzb_texture = nullptr;
    std::vector<WGPUTextureView> hzb_mip_views;      // all mips, for sampling
    std::vector<WGPUTextureView> hzb_storage_views;  // per-mip, for storage writes
    uint32_t        hzb_mip_count = 0;
    uint32_t        hzb_width = 0;
    uint32_t        hzb_height = 0;

    // Per-object uniforms in a storage buffer (indexed by instance obj_idx).
    WGPUBuffer        uniform_buffer    = nullptr;
    WGPUBindGroupLayout uniform_bgl     = nullptr;
    WGPUBindGroup     uniform_bind_group = nullptr;

    // Compute culling buffers (step 3: GPU-driven frustum cull).
    //   cull_aabb_buffer:    per-object world-space AABBs, 32 bytes/object
    //   cull_params_buffer:  mvp_proj_view mat4, 64 bytes
    //   cull_output_buffer:  visible count (1 uint), 4 bytes
    //   cull_indices_buffer: visible object uniform slots, 4 bytes/object
    WGPUBuffer cull_aabb_buffer    = nullptr;
    WGPUBuffer cull_params_buffer  = nullptr;
    WGPUBuffer cull_output_buffer  = nullptr;
    WGPUBuffer cull_indices_buffer = nullptr;
    WGPUBuffer cull_mesh_ids_buffer = nullptr;     // per-object mesh index, 4 bytes each
    WGPUBuffer cull_mesh_counts_buffer = nullptr;  // per-mesh atomic count, 4 bytes each
    WGPUBuffer cull_mesh_offsets_buffer = nullptr; // per-mesh base offset into buckets, 4 bytes each
    WGPUBuffer cull_mesh_buckets_buffer = nullptr; // flat array of object indices, grouped by mesh
    WGPUBuffer cull_indirect_args_buffer = nullptr; // per-mesh indirect draw args, 20 bytes each
    WGPUBuffer cull_mesh_index_counts_buffer = nullptr; // per-mesh index_count, 4 bytes each
    uint32_t   cull_aabb_capacity  = 0;   // in objects (aabb buffer size / 32)
    uint32_t   cull_indices_capacity = 0; // in objects
    uint32_t   cull_mesh_capacity = 0;    // in meshes

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

    // Per-mesh CPU-side metadata (parallel to scene_.meshes).
    // mesh_offsets[m] = base offset into cull_mesh_buckets_buffer for mesh m.
    // mesh_object_count[m] = total objects referencing mesh m (max instances).
    // Both are computed at load time and used by the draw loop to bind the
    // per-instance vertex buffer at the correct offset/size.
    std::vector<uint32_t>    mesh_offsets;
    std::vector<uint32_t>    mesh_object_count;

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

    // CSM shadow pipeline (depth-only, reads from shadow_uniform_buffer).
    WGPURenderPipeline  shadow_pipeline_       = nullptr;
    bool                shadow_pipeline_ready_ = false;
    WGPUBindGroupLayout shadow_uniform_bgl_    = nullptr;
    WGPUBindGroup       shadow_uniform_bg_     = nullptr;

    // Shadow cascade matrices, computed per frame (light-space view-proj
    // per cascade, snapped to texel grid for stability).
    // SSR composite (fullscreen passthrough for now).
    WGPURenderPipeline  ssr_composite_pipeline_  = nullptr;
    bool                ssr_composite_ready_     = false;
    WGPUBindGroupLayout ssr_scene_bgl_           = nullptr;
    WGPUBindGroup       ssr_scene_bg_            = nullptr;
    WGPUSampler         ssr_scene_sampler_       = nullptr;

    glm::mat4           shadow_view_proj_[4]   = {};
    // Cascade split ratios (near/far of the camera frustum that each cascade covers).
    // Computed once, uploaded to composite.

    // Composite shadow bind group (group 2): depth + shadow map + sampler + extra uniforms.
    WGPUBindGroupLayout composite_shadow_bgl_ = nullptr;
    WGPUBindGroup       composite_shadow_bg_  = nullptr;
    WGPUBuffer          composite_extra_buffer_ = nullptr;

    // Composite pipeline (reads G-buffer, writes final color).
    WGPURenderPipeline  composite_pipeline_ = nullptr;
    bool                composite_pipeline_ready_ = false;
    WGPUBindGroupLayout composite_bgl_ = nullptr;
    WGPUBindGroup       composite_bg_  = nullptr;

    // Material binding layout (group 1): base color + MR + normal + sampler.
    WGPUBindGroupLayout material_bgl_ = nullptr;

    // Compute culling pipeline + bind group (step 3).
    WGPUComputePipeline cull_pipeline_ = nullptr;
    WGPUBindGroupLayout cull_bgl_      = nullptr;
    WGPUBindGroup       cull_bg_       = nullptr;
    bool                cull_ready_    = false;

    // Indirect args builder (3c-1c-2a).
    WGPUComputePipeline indirect_build_pipeline_ = nullptr;
    WGPUBindGroupLayout indirect_build_bgl_      = nullptr;
    WGPUBindGroup       indirect_build_bg_       = nullptr;
    bool                indirect_build_ready_    = false;

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
