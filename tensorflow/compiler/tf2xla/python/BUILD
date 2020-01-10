load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_py_clif_cc",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_py_clif_cc(
    name = "xla_op_registry",
    srcs = ["xla_op_registry.clif"],
    pyclif_deps = [
        "//tensorflow/core/framework:kernel_def_pyclif",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
    ],
)

tf_custom_op_py_library(
    name = "xla",
    srcs = ["xla.py"],
    dso = ["//tensorflow/compiler/tf2xla/ops:_xla_ops.so"],
    kernels = [
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/compiler/xla:xla_data_proto_py",
    ],
)
