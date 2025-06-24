load("@bazel_skylib//rules:write_file.bzl", "write_file")
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
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

cc_library(
    name = "nccl_headers",
    hdrs = ["nccl.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
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

# This additional header allows us to determine the configured NCCL version
# without including the rest of NCCL.
write_file(
    name = "nccl_config_header",
    out = "nccl_config.h",
    content = [
        "#define TF_NCCL_VERSION \"%{nccl_version}\""
    ]
)

cc_library(
    name = "nccl_config",
    hdrs = ["nccl_config.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
)
