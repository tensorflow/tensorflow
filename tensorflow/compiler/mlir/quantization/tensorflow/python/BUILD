load("//tensorflow:tensorflow.bzl", "pytype_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/python:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "quantize_model_lib",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
    ],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite/quantization:quantization_config",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op_and_kernels",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = True,
)

cc_library(
    name = "quantize_model_cc",
    srcs = [
        "quantize_model_wrapper.cc",
    ],
    hdrs = [
        "quantize_model_wrapper.h",
    ],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",  # Required for pybind11
        "-parse_headers",
    ],
    visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:internal_visibility_allowlist_package",
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        ":quantize_model_lib",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op_and_kernels",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@pybind11",
    ],
    alwayslink = True,
)

tf_python_pybind_extension(
    name = "pywrap_quantize_model",
    srcs = [
        "pywrap_quantize_model.cc",
    ],
    hdrs = [
        "quantize_model_wrapper.h",
    ],
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_quantize_model",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types",
    ],
)

tf_py_test(
    name = "quantize_model_test",
    size = "medium",
    srcs = ["integration_test/quantize_model_test.py"],
    shard_count = 10,  # Parallelize the test to avoid timeouts.
    tags = ["no_pip"],
    deps = [
        ":quantize_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/saved_model:tag_constants",
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
