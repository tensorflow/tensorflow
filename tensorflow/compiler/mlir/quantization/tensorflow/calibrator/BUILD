load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "tf_kernel_library",
    "tf_py_strict_test",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

# Directly linked to `custom_aggregator_op`. In general, one should avoid directly depending on
# this target to avoid the ODR violation. Depend on `calibrator_singleton` instead.
cc_library(
    name = "calibrator_singleton_impl",
    srcs = ["calibrator_singleton.cc"],
    hdrs = ["calibrator_singleton.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        ":calibration_statistics_collector",
        ":calibration_statistics_proto_cc",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibrator_singleton",
    hdrs = ["calibrator_singleton.h"],
    compatible_with = get_compatible_with_portable(),
    deps = if_static([":calibrator_singleton_impl"]) + [
        ":calibration_statistics_collector",
        ":calibration_statistics_proto_cc",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "calibration_statistics_proto",
    srcs = ["calibration_statistics.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "calibration_statistics_py_pb2",
#     api_version = 2,
#     deps = [
#         ":calibration_statistics_proto",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "calibration_statistics_collector",
    srcs = ["calibration_statistics_collector.cc"],
    hdrs = ["calibration_statistics_collector.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_proto_cc",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "calibration_statistics_collector_test",
    size = "small",
    srcs = ["calibration_statistics_collector_test.cc"],
    deps = [
        ":calibration_statistics_collector",
        ":calibration_statistics_proto_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "calibrator_singleton_test",
    size = "small",
    srcs = ["calibrator_singleton_test.cc"],
    deps = [
        ":calibration_statistics_proto_cc",
        ":calibrator_singleton_impl",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_kernel_library(
    name = "custom_aggregator_op",
    srcs = ["custom_aggregator_op.cc"],
    compatible_with = get_compatible_with_portable(),
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
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    # Prevent unintentionally generating Python wrappers for all TF ops.
    op_allowlist = ["CustomAggregator"],
    py_lib_rule = py_strict_library,
    visibility = ["//visibility:private"],
    deps = [":custom_aggregator_op"],
)

tf_py_strict_test(
    name = "custom_aggregator_op_test",
    size = "small",
    srcs = ["integration_test/custom_aggregator_op_test.py"],
    deps = [
        ":calibration_statistics_proto_py",
        ":gen_custom_aggregator_op_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_quantize_model",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)
