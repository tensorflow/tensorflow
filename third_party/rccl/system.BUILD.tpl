filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rccl",
    srcs = ["librccl.so"],
    hdrs = ["rccl.h"],
    include_prefix = "third_party/rccl",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

genrule(
    name = "rccl-files",
    outs = [
        "librccl.so",
        "rccl.h",
    ],
    cmd = """cp "%{hdr_path}/rccl.h" "$(@D)/rccl.h" &&
           cp "%{install_path}/librccl.so" "$(@D)/librccl.so" """,
)
