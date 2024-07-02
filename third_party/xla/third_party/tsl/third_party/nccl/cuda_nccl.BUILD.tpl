licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "nccl",
    shared_library = "lib/libnccl.so.%{version}",
    hdrs = [":headers"],
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers", ":headers"],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/nccl",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
