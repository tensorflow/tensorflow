filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl",
    srcs = ["libnccl.so.2"],
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
        "libnccl.so.2",
        "nccl.h",
    ],
    cmd = """cp "/usr/include/nccl.h" "$(@D)/nccl.h" &&
           cp "/usr/lib/libnccl.so.2" "$(@D)/libnccl.so.2" """,
)
