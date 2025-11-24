load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)
load("//tensorflow/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//tensorflow/compiler/mlir/lite/...",
        "//tensorflow/compiler/mlir/quantization/...",
        "//tensorflow/compiler/mlir/tf2xla/transforms/...",
        "//tensorflow/lite/...",
        "//third_party/cloud_tpu/inference_converter/...",  # TPU Inference Converter V1
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//tensorflow:__pkg__",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "quantization",
    srcs = ["quantization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_quantization",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:save_model",
        "//tensorflow/core:protos_all_py",
    ],
)

# copybara:uncomment_begin(google-only)
# pytype_strict_library(
#     name = "quantize_model_test_base",
#     testonly = 1,
#     srcs = ["integration_test/quantize_model_test_base.py"],
#     tags = ["no_pip"],
#     visibility = ["//visibility:private"],
#     deps = [
#         "//third_party/py/mlir:ir",
#         "//third_party/py/mlir:stablehlo_dialect",
#         "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
#         "//third_party/py/numpy",
#         "//tensorflow:tensorflow_py",
#         "//tensorflow/compiler/mlir/stablehlo",
#         "//tensorflow/python/eager:def_function",
#         "//tensorflow/python/framework:dtypes",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/framework:tensor_spec",
#         "//tensorflow/python/module",
#         "//tensorflow/python/ops:array_ops",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python/ops:nn_ops",
#         "//tensorflow/python/ops:variables",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/platform:tf_logging",
#         "//tensorflow/python/saved_model:load",
#         "//tensorflow/python/saved_model:loader",
#         "//tensorflow/python/saved_model:save",
#         "//tensorflow/python/types:core",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# tf_py_strict_test(
#     name = "quantize_model_test",
#     srcs = ["integration_test/quantize_model_test.py"],
#     shard_count = 50,  # Parallelize the test to avoid timeouts.
#     deps = [
#         ":quantization",
#         ":quantize_model_test_base",
#         "//tensorflow/compiler/mlir/quantization/common/python:testing",
#         "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
#         "//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset",
#         "//tensorflow/python/eager:def_function",
#         "//tensorflow/python/framework:dtypes",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/framework:tensor_spec",
#         "//tensorflow/python/framework:test_lib",
#         "//tensorflow/python/module",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python/ops:nn_ops",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/saved_model:load",
#         "//tensorflow/python/saved_model:save",
#         "//tensorflow/python/saved_model:tag_constants",
#         "//tensorflow/python/types:core",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
# copybara:uncomment_end

# This is a header-only target. The purpose of `pywrap_quantization_lib_*` targets is to expose only
# the symbols that are required by `pywrap_quantization` that translates them to python functions.
# The only intended use case of this library is by `pywrap_quantization`. Not letting
# `pywrap_quantization` directly depend on sub-libraries like `static_range_srq` and instead haiving
# a consolidated impl library `pywrap_quantization_lib_impl` allows the maintainers to avoid
# declaring multiple impl libraries to `libtensorflow_cc` and `lib_pywrap_tensorflow_internal`,
# which is required to avoid ODR violations.
cc_library(
    name = "pywrap_quantization_lib_header_only",
    srcs = [],
    hdrs = ["pywrap_quantization_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],  # ONLY for `pywrap_quantization`.
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# See the comments for `pywrap_quantization_lib_header_only`.
cc_library(
    name = "pywrap_quantization_lib_impl",
    srcs = ["pywrap_quantization_lib.cc"],
    hdrs = ["pywrap_quantization_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow:__pkg__",  # For libtensorflow_cc.so.
        "//tensorflow/python:__pkg__",  # For lib_pywrap_tensorflow_internal.so.
    ],
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:config",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:static_range_ptq",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:weight_only_ptq",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_quantization",
    srcs = ["pywrap_quantization.cc"],
    pytype_srcs = ["pywrap_quantization.pyi"],
    starlark_only = True,
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    # Each dependency MUST be either header-only or exclusive.
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:type_casters",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(
        if_false = [":pywrap_quantization_lib_header_only"],
        if_true = [":pywrap_quantization_lib_impl"],
    ),
)
