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
    name = "nccl_shared_library",
    shared_library = "lib/libnccl.so.%{libnccl_version}",
    hdrs = [":headers"],
    deps = ["@local_config_cuda//cuda:cuda_headers", ":headers"],
)
%{multiline_comment}
cc_library(
    name = "nccl",
    %{comment}deps = [":nccl_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/nccl/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/nccl*.h",
    %{comment}]),
    include_prefix = "third_party/nccl",
    includes = ["include/"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
