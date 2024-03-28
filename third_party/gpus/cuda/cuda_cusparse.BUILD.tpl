licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "include",
    srcs = glob([
        "include/**",
    ]),
)

cc_import(
    name = "cusparse",
    hdrs = [":headers"],
    shared_library = "lib/libcusparse.so.%{version}",
    linkopts = ["-lgomp"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cusparse_lib",
    srcs = ["lib/libcusparse.so.%{version}"],
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
