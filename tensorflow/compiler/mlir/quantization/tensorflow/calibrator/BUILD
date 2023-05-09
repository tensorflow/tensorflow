load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_cloud",
    "tf_kernel_library",
    "tf_py_test",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

# Directly linked to `custom_aggregator_op`. In general, one should avoid directly depending on
# this target to avoid the ODR violation. Depend on `calibrator_singleton` instead.
cc_library(
    name = "calibrator_singleton_impl",
    srcs = ["calibrator_singleton.cc"],
    hdrs = ["calibrator_singleton.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "calibrator_singleton",
    hdrs = ["calibrator_singleton.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = if_static([":calibrator_singleton_impl"]) + [
        "@com_google_absl//absl/container:flat_hash_map",
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
        ":calibrator_singleton_impl",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "custom_aggregator_op",
    srcs = ["custom_aggregator_op.cc"],
    compatible_with = get_compatible_with_cloud(),
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:__pkg__",
    ],
    deps = [
        ":calibrator_singleton_impl",
        "//tensorflow/core:framework",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_custom_aggregator_op_wrapper",
    out = "custom_aggregator_op_wrapper.py",
    # Prevent unintentionally generating Python wrappers for all TF ops.
    op_allowlist = ["CustomAggregator"],
    visibility = ["//visibility:private"],
    deps = [":custom_aggregator_op"],
)

tf_py_test(
    name = "custom_aggregator_op_test",
    size = "small",
    srcs = ["integration_test/custom_aggregator_op_test.py"],
    tags = ["no_pip"],
    deps = [
        ":gen_custom_aggregator_op_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_quantize_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)
