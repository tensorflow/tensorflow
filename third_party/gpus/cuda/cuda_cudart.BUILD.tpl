licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "static",
    srcs = ["lib/libcudart_static.a"],
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)

cc_import(
    name = "cuda_driver",
    shared_library = "lib/stubs/libcuda.so",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cudart",
    hdrs = [":headers"],
    shared_library = "lib/libcudart.so.%{version}",
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
