# Description:
#   A library for decoding and encoding GIF images

licenses(["notice"])  # MIT

cc_library(
    name = "gif",
    srcs = [
        "dgif_lib.c",
        "egif_lib.c",
        "gif_err.c",
        "gif_font.c",
        "gif_hash.c",
        "gif_hash.h",
        "gif_lib_private.h",
        "gifalloc.c",
        "openbsd-reallocarray.c",
        "quantize.c",
    ],
    hdrs = ["gif_lib.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = select({
        ":windows": [":windows_polyfill"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "windows_polyfill",
    hdrs = ["windows/unistd.h"],
    includes = ["windows"],
)

genrule(
    name = "windows_unistd_h",
    outs = ["windows/unistd.h"],
    cmd = "touch $@",
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows_msvc"},
)
