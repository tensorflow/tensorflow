licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "nvshmem_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libnvshmem_host.so.%{libnvshmem_host_version}",
)
%{multiline_comment}
cc_library(
    name = "nvshmem",
    %{comment}deps = [":nvshmem_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/nvshmem/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/**", 
    %{comment}]),
    include_prefix = "third_party/nvshmem",
    includes = ["include"],
    strip_include_prefix = "include",
)