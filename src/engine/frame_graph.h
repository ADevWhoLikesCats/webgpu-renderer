#ifndef PHOTONIX_FRAME_GRAPH_H
#define PHOTONIX_FRAME_GRAPH_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu.h>

namespace engine {

using PxResourceId = uint32_t;
constexpr PxResourceId PX_RESOURCE_INVALID = 0xFFFFFFFFu;

struct FgResource {
    std::string name;
    WGPUTextureView view = nullptr;  // for pass bind-group resolution
};

struct FgPass {
    std::string name;
    std::vector<PxResourceId> reads;
    std::vector<PxResourceId> writes;
    std::function<void(WGPUCommandEncoder)> execute;
};

class FrameGraph {
public:
    // Register a resource by name. Engine owns the underlying texture/view;
    // the graph only tracks handles.
    PxResourceId register_resource(const char* name, WGPUTextureView view);

    // Look up a resource's view by id. Returns nullptr for invalid ids.
    WGPUTextureView view(PxResourceId id) const;

    // Add a pass. Passes execute in registration order for v1.
    void add_pass(const char* name,
                  std::vector<PxResourceId> reads,
                  std::vector<PxResourceId> writes,
                  std::function<void(WGPUCommandEncoder)> execute);

    // Execute all registered passes on the given encoder in registration order.
    void execute(WGPUCommandEncoder encoder);

    // Clear the pass list for the next frame. Resources persist.
    void reset_passes();

    // Clear everything (shutdown).
    void clear();

private:
    std::vector<FgResource> resources_;
    std::vector<FgPass>     passes_;
};

} // namespace engine

#endif // PHOTONIX_FRAME_GRAPH_H
