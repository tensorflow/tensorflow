licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

exports_files([
    "version.txt",
])
%{multiline_comment}
cc_import(
    name = "curand_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcurand.so.%{libcurand_version}",
)
%{multiline_comment}
cc_library(
    name = "curand",
    %{comment}deps = [":curand_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/curand/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob(["include/curand*.h"]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
