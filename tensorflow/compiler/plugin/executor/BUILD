licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "plugin_lib",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
