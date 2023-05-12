load("//tensorflow:pytype.default.bzl", "pytype_library", "pytype_strict_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_cloud",
    "tf_py_test",
    "tf_python_pybind_extension",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/python:__subpackages__",
    ],
    licenses = ["notice"],
)

# Do NOT directly depend on `quantize_model_cc_impl` unless it is necessary
# (i.e. undefined symbol). See the comments in `quantize_model_cc`.
cc_library(
    name = "quantize_model_cc_impl",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = [
        # Directly linked to `libtensorflow_cc.so` or
        # `_pywrap_tensorflow_internal.so` if static build.
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op",  # Required for CustomAggregator op registration.
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:convert_asset_args",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:save_variables",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:status_macro",
        "//tensorflow/compiler/mlir/quantization/tensorflow/debugging:mlir_dump",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
    ],
)

# OSS: This is a header-only target. The implementation target `quantize_model_cc_impl` is
# directly linked to `lib_pywrap_tensorflow_internal.so`, so in most use cases of python-
# exported symbols depending directly on `quantize_model_cc_impl` should be unnecessary.
# Using the header-only target will help avoid the ODR violation.
cc_library(
    name = "quantize_model_cc",
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = if_static([":quantize_model_cc_impl"]) + [
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Exports python symbols via pybind11.
tf_python_pybind_extension(
    name = "pywrap_quantize_model",
    srcs = ["pywrap_quantize_model.cc"],
    # All deps must be header-only.
    deps = [
        ":quantize_model_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

tf_py_test(
    name = "pywrap_quantize_model_test",
    srcs = [
        "pywrap_quantize_model_test.py",
    ],
    tags = ["no_pip"],
    deps = [
        ":pywrap_quantize_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "save_model",
    srcs = [
        "save_model.py",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:training_lib",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_quantize_model",
        ":representative_dataset",
        ":save_model",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "quantize_model_test",
    size = "medium",
    srcs = ["integration_test/quantize_model_test.py"],
    shard_count = 50,  # Parallelize the test to avoid timeouts.
    tags = ["no_pip"],
    deps = [
        ":quantize_model",
        ":quantize_model_test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:tag_constants",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_library(
    name = "quantize_model_test_base",
    testonly = 1,
    srcs = ["integration_test/quantize_model_test_base.py"],
    tags = ["no_pip"],
    deps = [
        ":representative_dataset",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "concurrency_test",
    size = "medium",
    srcs = ["integration_test/concurrency_test.py"],
    tags = ["no_pip"],
    deps = [
        ":quantize_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/saved_model:tag_constants",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "representative_dataset",
    srcs = [
        "representative_dataset.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
    ],
)

tf_py_test(
    name = "representative_dataset_test",
    srcs = ["representative_dataset_test.py"],
    tags = ["no_pip"],
    deps = [
        ":representative_dataset",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
    ],
)
