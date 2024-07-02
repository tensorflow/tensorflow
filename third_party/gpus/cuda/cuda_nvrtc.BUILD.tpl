licenses(["restricted"])  # NVIDIA proprietary license

cc_import(
    name = "nvrtc_main",
    hdrs = [":headers"],
    shared_library = "lib/libnvrtc.so.%{version}",
)

cc_import(
    name = "nvrtc_builtins",
    hdrs = [":headers"],
    shared_library = "lib/libnvrtc-builtins.so.%{major_minor_version}",
)

cc_library(
    name = "nvrtc",
    deps = [
        ":nvrtc_main",
        ":nvrtc_builtins",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
