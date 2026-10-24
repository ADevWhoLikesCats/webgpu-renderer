cmake_minimum_required(VERSION 3.24)
project(photonix LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Third-party headers (vendored) ----
set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(NOT EXISTS ${THIRD_PARTY}/SDL3/include/SDL3/SDL.h)
    message(FATAL_ERROR "Missing vendored SDL3 headers")
endif()
if(NOT EXISTS ${THIRD_PARTY}/dawn/include/webgpu/webgpu.h)
    message(FATAL_ERROR "Missing vendored Dawn headers")
endif()

# ---- tinygltf (header-only via FetchContent) ----
include(FetchContent)
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        release
    GIT_SHALLOW    TRUE
)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL     OFF CACHE BOOL "" FORCE)
set(TINYGLTF_USE_CPP14   ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# ---- Threads (required by Dawn's exported interface) ----
find_package(Threads REQUIRED)

# ---- Dawn ----
find_package(Dawn CONFIG REQUIRED)
if(NOT TARGET dawn::webgpu_dawn)
    message(FATAL_ERROR "dawn::webgpu_dawn missing at ${Dawn_DIR}")
endif()
message(STATUS "Dawn: ${Dawn_DIR}")

# ---- SDL3 ----
find_package(SDL3 CONFIG REQUIRED)

# ---- Bullet (module mode) ----
find_package(Bullet REQUIRED)
if(Bullet_FOUND AND NOT TARGET Bullet::Bullet)
    add_library(Bullet::Bullet INTERFACE IMPORTED)
    set_target_properties(Bullet::Bullet PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${BULLET_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${BULLET_LIBRARIES}")
endif()

# ---- Wayland client (for the raw wl_surface* we receive) ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(WAYLAND REQUIRED wayland-client)

# ---- glm (fetch, header-only) ----
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# =========================================================
# photonix — the library
# =========================================================
add_library(photonix STATIC
    src/engine/api.cpp
    src/engine/engine.cpp
    src/engine/frame_graph.cpp
    src/sdl3webgpu.c
)

target_compile_definitions(photonix PRIVATE ENGINE_BUILDING)

target_include_directories(photonix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
        ${THIRD_PARTY}/SDL3/include
        ${THIRD_PARTY}/dawn/include
        ${WAYLAND_INCLUDE_DIRS}
)

target_link_libraries(photonix PRIVATE
    dawn::webgpu_dawn
    SDL3::SDL3
    Bullet::Bullet
    glm::glm
    ${WAYLAND_LIBRARIES}
)

if(NOT TARGET tinygltf AND NOT TARGET tinygltf::tinygltf)
    target_include_directories(photonix PRIVATE ${tinygltf_SOURCE_DIR})
endif()
# (stb not needed; we write PPM manually)

# =========================================================
# engine_demo — the executable
# =========================================================
add_executable(engine_demo apps/demo/main.cpp)
target_link_libraries(engine_demo PRIVATE photonix SDL3::SDL3)
