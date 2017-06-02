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
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@protobuf//:protobuf_headers",
    ],
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
