load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load(
    "//tensorflow:tensorflow.default.bzl",
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
    deps = [
        ":pywrap_quantization",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:save_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/saved_model:loader",
    ],
)

pytype_strict_library(
    name = "quantize_model_test_base",
    testonly = 1,
    srcs = ["integration_test/quantize_model_test_base.py"],
    tags = ["no_pip"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "quantize_model_test",
    srcs = ["integration_test/quantize_model_test.py"],
    deps = [
        ":quantization",
        ":quantize_model_test_base",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:representative_dataset",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:tag_constants",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_quantization",
    srcs = ["pywrap_quantization.cc"],
    pytype_srcs = ["pywrap_quantization.pyi"],
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:debugger",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:io",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:assign_ids",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:statistics",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:type_casters",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
