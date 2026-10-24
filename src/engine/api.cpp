#include "engine_internal.h"
#include "engine/engine.h"

using engine::Engine;

extern "C" {

engine_result engine_renderer_create(const engine_renderer_desc* desc,
                                     const engine_window_desc*   window,
                                     engine_renderer**           out)
{
    if (!out || !window) return ENGINE_ERR_INVALID_ARG;
    *out = nullptr;

    auto* e = new (std::nothrow) Engine();
    if (!e) return ENGINE_ERR_INTERNAL;

    const char* app = (desc && desc->app_name) ? desc->app_name : "photonix";
    int validation  = (desc && desc->enable_validation) ? 1 : 0;
    int software    = (desc && desc->prefer_software_device) ? 1 : 0;

    void* native  = nullptr;
    void* display = nullptr;
    if (window->kind == ENGINE_WINDOW_WAYLAND) {
        native  = window->native_handle;
        display = window->display_handle;
    } else { delete e; return ENGINE_ERR_UNSUPPORTED; }

    std::string err;
    bool ok = e->init(app, validation, software, native, display,
                      window->width, window->height, err);
    if (!ok) {
        if (!err.empty()) std::fprintf(stderr, "[engine] init: %s\n", err.c_str());
        delete e;
        return ENGINE_ERR_NO_BACKEND;
    }
    *out = reinterpret_cast<engine_renderer*>(e);
    return ENGINE_OK;
}

void engine_renderer_destroy(engine_renderer* r) {
    delete reinterpret_cast<Engine*>(r);
}

engine_result engine_renderer_get_adapter_info(engine_renderer* r, engine_adapter_info* out) {
    if (!r || !out) return ENGINE_ERR_INVALID_ARG;
    reinterpret_cast<Engine*>(r)->get_adapter_info(
        out->device_name, sizeof(out->device_name),
        out->description, sizeof(out->description),
        &out->backend_type, &out->vendor_id, &out->device_id);
    return ENGINE_OK;
}

engine_result engine_renderer_resize(engine_renderer* r, uint32_t w, uint32_t h) {
    if (!r) return ENGINE_ERR_INVALID_ARG;
    reinterpret_cast<Engine*>(r)->resize(w, h);
    return ENGINE_OK;
}

engine_result engine_renderer_begin_frame(engine_renderer* r) {
    if (!r) return ENGINE_ERR_INVALID_ARG;
    reinterpret_cast<Engine*>(r)->begin_frame();
    return ENGINE_OK;
}

engine_result engine_renderer_end_frame(engine_renderer* r) {
    if (!r) return ENGINE_ERR_INVALID_ARG;
    reinterpret_cast<Engine*>(r)->end_frame();
    return ENGINE_OK;
}

void engine_renderer_tick(engine_renderer* r) {
    if (r) reinterpret_cast<Engine*>(r)->tick();
}

engine_result engine_renderer_load_gltf(engine_renderer* r, const char* path) {
    if (!r || !path) return ENGINE_ERR_INVALID_ARG;
    std::string err;
    if (!reinterpret_cast<Engine*>(r)->load_gltf(path, err)) {
        if (!err.empty()) std::fprintf(stderr, "[engine] gltf: %s\n", err.c_str());
        return ENGINE_ERR_INTERNAL;
    }
    return ENGINE_OK;
}

const char* engine_version_string(void) { return "photonix-engine 0.1.0"; }

} // extern "C"
