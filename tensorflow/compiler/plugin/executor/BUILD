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
        "//third_party/eigen3",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service:hlo_evaluator",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "@protobuf//:protobuf_headers",
        "@local_config_cuda//cuda:cuda_headers",
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

