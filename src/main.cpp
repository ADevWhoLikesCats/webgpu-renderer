#include <cstdio>
#include <SDL3/SDL.h>

int main(int, char**) {
    // Force native Wayland BEFORE SDL_Init. XWayland is the broken path.
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const char* drv = SDL_GetCurrentVideoDriver();
    std::printf("SDL video driver: %s\n", drv ? drv : "(null)");
    if (!drv || SDL_strcmp(drv, "wayland") != 0) {
        std::fprintf(stderr, "FATAL: not on native Wayland; got '%s'\n",
                     drv ? drv : "null");
        SDL_Quit();
        return 2;
    }

    SDL_Window* win = SDL_CreateWindow("MyRenderer - window check", 1280, 720,
                                       SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    std::printf("Window flags: 0x%llx\n",
                (unsigned long long)SDL_GetWindowFlags(win));

    SDL_Renderer* ren = SDL_CreateRenderer(win, nullptr);
    if (!ren) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE)
                running = false;
        }
        SDL_SetRenderDrawColor(ren, 0x20, 0x60, 0xC0, 0xFF);
        SDL_RenderClear(ren);
        SDL_RenderPresent(ren);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
