licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "cupti",
    hdrs = [":headers"],
    shared_library = "lib/libcupti.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/extras/CUPTI/include",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
