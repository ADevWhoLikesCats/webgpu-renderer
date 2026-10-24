#include <cstdio>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "sdl3webgpu.h"               // from the vendored shim
#include <webgpu/webgpu.h>            // from third_party/dawn/include

#include <bullet/btBulletDynamicsCommon.h>
#include <tiny_gltf.h>                 // from FetchContent

int main() {
    std::printf("SDL version: %d\n", SDL_VERSION);
    std::printf("tinygltf: header reachable\n");
    return 0;
}
