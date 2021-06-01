# Description:
# CLIF wrappers for TensorFlow SavedModels.

load("//tensorflow/core/platform:build_config.bzl", "tf_py_clif_cc")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_py_clif_cc(
    name = "loader",
    srcs = ["loader.clif"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
    ],
)
