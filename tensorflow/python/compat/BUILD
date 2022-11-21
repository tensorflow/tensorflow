load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "v2_compat",
    srcs = ["v2_compat.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tf2",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/experimental/ops:counter",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:random_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "compat_test",
    size = "small",
    srcs = ["compat_test.py"],
    tags = ["nofwdcompat"],
    deps = [
        ":compat",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
    ],
)

tf_py_test(
    name = "disable_v2_behavior_test",
    size = "small",
    srcs = ["disable_v2_behavior_test.py"],
    deps = [
        ":v2_compat",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)
