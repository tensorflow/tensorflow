load("//tensorflow:pytype.default.bzl", "pytype_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_library(
    name = "ops",
    srcs = ["tpu_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":gen_tpu_embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python/tpu:tpu_function",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_tpu_embedding_ops",
    out = "gen_tpu_embedding_ops.py",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/ops:tpu_configuration_ops_op_lib",
        "//tensorflow/core/ops:tpu_embedding_ops_op_lib",
        "//tensorflow/core/tpu/ops:tpu_embedding_ops",
    ],
)
