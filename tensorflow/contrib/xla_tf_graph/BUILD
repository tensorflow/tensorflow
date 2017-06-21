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
    name = "xla_tf_graph_util",
    srcs = [
        "xla_tf_graph_util.cc",
    ],
    hdrs = [
        "xla_tf_graph_util.h",
    ],
    deps = [
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
    name = "xla_tf_graph_util_test",
    size = "small",
    srcs = ["xla_tf_graph_util_test.cc"],
    linkstatic = 1,
    deps = [
        ":xla_tf_graph_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit:xla_cpu_jit",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:cwise_op",
    ],
)
