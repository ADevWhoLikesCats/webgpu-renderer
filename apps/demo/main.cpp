#include <cstdio>
#include <SDL3/SDL.h>
#include <SDL3/SDL_properties.h>
#include <engine/engine.h>

int main(int, char**) {
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
    if (!SDL_Init(SDL_INIT_VIDEO)) { std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }

    const char* drv = SDL_GetCurrentVideoDriver();
    std::printf("demo: SDL driver = %s\n", drv ? drv : "(null)");
    if (!drv || SDL_strcmp(drv, "wayland") != 0) { SDL_Quit(); return 2; }

    SDL_Window* win = SDL_CreateWindow("photonix demo", 1280, 720,
                                       SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) { std::fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); SDL_Quit(); return 1; }

    // SDL3 exposes the wayland surface via a window property.
    SDL_PropertiesID props = SDL_GetWindowProperties(win);
    void* wl_surface = SDL_GetPointerProperty(
        props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    if (!wl_surface) {
        std::fprintf(stderr, "demo: no wl_surface (SDL not on native Wayland)\n");
        SDL_DestroyWindow(win); SDL_Quit(); return 1;
    }

    // Retrieve the global wl_display pointer from SDL
    void* wl_display = SDL_GetPointerProperty(
        SDL_GetGlobalProperties(),
        SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER,
        nullptr);
    if (!wl_display) {
        std::fprintf(stderr, "demo: no wl_display pointer\n");
        SDL_DestroyWindow(win); SDL_Quit(); return 1;
    }

    int ww = 0, wh = 0;
    SDL_GetWindowSizeInPixels(win, &ww, &wh);

    engine_renderer_desc rdesc = {};
    rdesc.app_name              = "photonix-demo";
    rdesc.app_version           = 1;
    rdesc.enable_validation     = 1;
    rdesc.prefer_software_device = 1; /* llvmpipe; flip to 0 to test ANV */

    engine_window_desc wdesc = {};
    wdesc.kind          = ENGINE_WINDOW_WAYLAND;
    wdesc.native_handle = wl_surface;
    wdesc.display_handle = wl_display;   // new field
    wdesc.width         = (uint32_t)ww;
    wdesc.height        = (uint32_t)wh;
    wdesc.dpi_scale     = 1.0f;

    engine_renderer* r = nullptr;
    if (engine_renderer_create(&rdesc, &wdesc, &r) != ENGINE_OK) {
        std::fprintf(stderr, "engine_renderer_create failed\n");
        SDL_DestroyWindow(win); SDL_Quit(); return 1;
    }

    engine_adapter_info ai = {};
    engine_renderer_get_adapter_info(r, &ai);
    std::printf("demo: adapter=%s backend=%u\n", ai.device_name, ai.backend_type);

    // Load a car model. Path is relative to the current working directory.
    if (engine_renderer_load_gltf(r, "assets/car.glb") != ENGINE_OK) {
        std::fprintf(stderr, "demo: gltf load failed\n");
    }

    bool running = true;
    Uint64 t0 = SDL_GetTicks();
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) running = false;
            if (e.type == SDL_EVENT_WINDOW_RESIZED) {
                int nw = 0, nh = 0;
                SDL_GetWindowSizeInPixels(win, &nw, &nh);
                engine_renderer_resize(r, (uint32_t)nw, (uint32_t)nh);
            }
        }
        engine_renderer_tick(r);
        engine_renderer_begin_frame(r);
        engine_renderer_end_frame(r);
        if (SDL_GetTicks() - t0 > 15000) running = false; /* smoke timeout */
        SDL_Delay(16);
    }

    engine_renderer_destroy(r);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
