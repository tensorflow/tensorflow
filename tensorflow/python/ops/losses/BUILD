load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":losses_impl",
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:confusion_matrix",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "losses_impl",
    srcs = ["losses_impl.py"],
    strict_deps = True,
    deps = [
        ":util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:confusion_matrix",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:weights_broadcast_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    strict_deps = True,
    deps = [
        ":util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)
