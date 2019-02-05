# Description:
# CLIF wrappers for TensorFlow SavedModels.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

load("//tensorflow/core:platform/default/build_config.bzl", "tf_py_clif_cc")

tf_py_clif_cc(
    name = "loader",
    srcs = ["loader.clif"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
    ],
)
