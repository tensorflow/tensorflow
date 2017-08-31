# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

cc_library(
    name = "xlagen_lib",
    srcs = [
        "xlagen.cc",
    ],
    hdrs = [
        "xlagen.h",
    ],
    deps = [
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/client",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "xlagen_test",
    size = "small",
    srcs = ["xlagen_test.cc"],
    linkstatic = 1,
    deps = [
        ":xlagen_lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:ops",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
