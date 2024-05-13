load("//tensorflow:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "tf_kernel_library",
    "tf_py_strict_test",
)
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/core:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "calibration_statistics_collector_base",
    hdrs = ["calibration_statistics_collector_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_min_max",
    srcs = ["calibration_statistics_collector_min_max.cc"],
    hdrs = ["calibration_statistics_collector_min_max.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_average_min_max",
    srcs = ["calibration_statistics_collector_average_min_max.cc"],
    hdrs = ["calibration_statistics_collector_average_min_max.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_histogram",
    srcs = ["calibration_statistics_collector_histogram.cc"],
    hdrs = ["calibration_statistics_collector_histogram.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

pytype_strict_library(
    name = "calibration_algorithm",
    srcs = ["calibration_algorithm.py"],
    deps = [
        ":calibration_statistics_proto_py",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "calibration_algorithm_test",
    srcs = ["calibration_algorithm_test.py"],
    deps = [
        ":calibration_algorithm",
        ":calibration_statistics_proto_py",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
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

tf_cc_test(
    name = "calibration_statistics_collector_test",
    size = "small",
    srcs = ["calibration_statistics_collector_test.cc"],
    deps = [
        ":calibration_statistics_collector_average_min_max",
        ":calibration_statistics_collector_histogram",
        ":calibration_statistics_collector_min_max",
        ":calibration_statistics_proto_cc",
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
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:errors",
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
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_kernel_library(
    name = "calibration_statistics_saver_op",
    srcs = ["calibration_statistics_saver_op.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:__pkg__",
    ],
    deps = [
        ":calibration_statistics_collector_average_min_max",
        ":calibration_statistics_collector_base",
        ":calibration_statistics_collector_histogram",
        ":calibration_statistics_collector_min_max",
        ":calibration_statistics_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
    ],
)

tf_cc_test(
    name = "calibration_statistics_saver_op_test",
    srcs = ["calibration_statistics_saver_op_test.cc"],
    deps = [
        ":calibration_statistics_proto_cc",
        ":calibration_statistics_saver_op",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)
