load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_py_test",
)

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "calibrator_singleton",
    srcs = ["calibrator_singleton.cc"],
    hdrs = ["calibrator_singleton.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "calibrator_singleton_test",
    size = "small",
    srcs = ["calibrator_singleton_test.cc"],
    deps = [
        ":calibrator_singleton",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "custom_aggregator_op_and_kernels",
    srcs = ["custom_aggregator_op.cc"],
    copts = tf_copts(),
    deps = [
        ":calibrator_singleton",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "_custom_aggregator_op.so",
    srcs = ["custom_aggregator_op.cc"],
    deps = [
        ":calibrator_singleton",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_custom_aggregator_op_wrapper",
    out = "custom_aggregator_op_wrapper.py",
    deps = [":custom_aggregator_op_and_kernels"],
)

tf_custom_op_py_library(
    name = "custom_aggregator_op",
    srcs = ["custom_aggregator_op.py"],
    dso = [":_custom_aggregator_op.so"],
    kernels = [":custom_aggregator_op_and_kernels"],
    srcs_version = "PY3",
    deps = [
        ":gen_custom_aggregator_op_wrapper",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "custom_aggregator_op_test",
    size = "small",
    srcs = ["custom_aggregator_op_test.py"],
    tags = [
        "no_oss",  # TODO(b/220688154): Enable OSS tests.
    ],
    deps = [
        ":custom_aggregator_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_wrapper",
    ],
)
