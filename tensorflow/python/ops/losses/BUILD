load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//engedu/ml/tf_from_scratch:__pkg__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
        ":losses_impl",
        ":util",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "losses_impl",
    srcs = ["losses_impl.py"],
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":losses",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)
