#ifndef ENGINE_H
#define ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENGINE_API_VERSION 1

#if defined(_WIN32)
#  ifdef ENGINE_BUILDING
#    define ENGINE_API __declspec(dllexport)
#  else
#    define ENGINE_API __declspec(dllimport)
#  endif
#else
#  ifdef ENGINE_BUILDING
#    define ENGINE_API __attribute__((visibility("default")))
#  else
#    define ENGINE_API
#  endif
#endif

/* Opaque handles. */
typedef struct engine_renderer engine_renderer;
typedef struct engine_texture  engine_texture;

/* Status codes. */
typedef enum {
    ENGINE_OK              = 0,
    ENGINE_ERR_INVALID_ARG = 1,
    ENGINE_ERR_NO_BACKEND  = 2,
    ENGINE_ERR_DEVICE_LOST = 3,
    ENGINE_ERR_UNSUPPORTED = 4,
    ENGINE_ERR_INTERNAL    = 5
} engine_result;

/* Native window kinds the library accepts. */
typedef enum {
    ENGINE_WINDOW_NONE    = 0,
    ENGINE_WINDOW_WAYLAND = 1,
    ENGINE_WINDOW_X11     = 2,
    ENGINE_WINDOW_WIN32   = 3,
    ENGINE_WINDOW_COCOA   = 4
} engine_window_kind;

typedef struct {
    engine_window_kind kind;
    void*              native_handle;
    void*              display_handle;   /* Wayland: wl_display*; X11: Display* */
    uint32_t           width;
    uint32_t           height;
    float              dpi_scale;
} engine_window_desc;

typedef struct {
    const char* app_name;
    uint32_t    app_version;
    int         enable_validation;
    int         prefer_software_device;
} engine_renderer_desc;

typedef struct {
    char     device_name[256];
    char     description[256];
    uint32_t backend_type;
    uint32_t vendor_id;
    uint32_t device_id;
} engine_adapter_info;

/* Lifecycle. */
ENGINE_API engine_result engine_renderer_create(const engine_renderer_desc* desc,
                                                const engine_window_desc*   window,
                                                engine_renderer**           out);
ENGINE_API void          engine_renderer_destroy(engine_renderer* r);

/* Info. */
ENGINE_API engine_result engine_renderer_get_adapter_info(engine_renderer* r,
                                                          engine_adapter_info* out);

/* Window / surface. */
ENGINE_API engine_result engine_renderer_resize(engine_renderer* r,
                                                uint32_t width, uint32_t height);

/* Frame. Host drives the loop; library never blocks. */
ENGINE_API engine_result engine_renderer_begin_frame(engine_renderer* r);
ENGINE_API engine_result engine_renderer_end_frame(engine_renderer* r);
ENGINE_API void          engine_renderer_tick(engine_renderer* r);

/* Scene. UTF-8 path, no STL on the boundary. */
ENGINE_API engine_result engine_renderer_load_gltf(engine_renderer* r,
                                                   const char* utf8_path);

/* Version. */
ENGINE_API const char* engine_version_string(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENGINE_H */
