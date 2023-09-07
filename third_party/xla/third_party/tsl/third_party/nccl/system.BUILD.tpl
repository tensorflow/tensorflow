load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags"
)

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl",
    srcs = ["libnccl.so.%{nccl_version}"],
    hdrs = ["nccl.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
    linkopts = cuda_rpath_flags("nvidia/nccl/lib"),
)

genrule(
    name = "nccl-files",
    outs = [
        "libnccl.so.%{nccl_version}",
        "nccl.h",
    ],
    cmd = """
cp "%{nccl_header_dir}/nccl.h" "$(@D)/nccl.h" &&
cp "%{nccl_library_dir}/libnccl.so.%{nccl_version}" \
  "$(@D)/libnccl.so.%{nccl_version}"
""",
)
