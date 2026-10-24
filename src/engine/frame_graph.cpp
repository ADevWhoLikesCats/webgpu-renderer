#include "frame_graph.h"

#include <cstdio>

namespace engine {

PxResourceId FrameGraph::register_resource(const char* name, WGPUTextureView view) {
    resources_.push_back({ name ? name : "?", view });
    return static_cast<PxResourceId>(resources_.size() - 1);
}

WGPUTextureView FrameGraph::view(PxResourceId id) const {
    if (id == PX_RESOURCE_INVALID || id >= resources_.size()) return nullptr;
    return resources_[id].view;
}

void FrameGraph::add_pass(const char* name,
                          std::vector<PxResourceId> reads,
                          std::vector<PxResourceId> writes,
                          std::function<void(WGPUCommandEncoder)> execute)
{
    FgPass p;
    p.name    = name ? name : "?";
    p.reads   = std::move(reads);
    p.writes  = std::move(writes);
    p.execute = std::move(execute);
    passes_.push_back(std::move(p));
}

void FrameGraph::execute(WGPUCommandEncoder encoder) {
    for (auto& p : passes_) {
        if (p.execute) p.execute(encoder);
    }
}

void FrameGraph::reset_passes() {
    passes_.clear();
}

void FrameGraph::clear() {
    passes_.clear();
    resources_.clear();
}

} // namespace engine
