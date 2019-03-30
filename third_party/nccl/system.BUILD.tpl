filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl",
    srcs = ["libnccl.so.%{version}"],
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
        "libnccl.so.%{version}",
        "nccl.h",
    ],
    cmd = """cp "%{hdr_path}/nccl.h" "$(@D)/nccl.h" &&
           cp "%{install_path}/libnccl.so.%{version}" "$(@D)/libnccl.so.%{version}" """,
)
