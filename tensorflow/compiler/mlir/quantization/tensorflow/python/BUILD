load("//tensorflow:tensorflow.bzl", "pytype_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization:__subpackages__",
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
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_python_pybind_extension(
    name = "quantize_model_wrapper",
    srcs = [
        "quantize_model_wrapper.cc",
    ],
    module_name = "quantize_model_wrapper",
    deps = [
        ":quantize_model_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
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
        ":quantize_model_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/saved_model:utils",
        "@flatbuffers//:runtime_py",
    ],
)

tf_py_test(
    name = "quantize_model_test",
    size = "medium",
    srcs = ["integration_test/quantize_model_test.py"],
    tags = [
        "no_oss",  # TODO(b/220688154): Enable OSS tests.
    ],
    deps = [
        ":quantize_model",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "concurrency_test",
    size = "medium",
    srcs = ["integration_test/concurrency_test.py"],
    tags = [
        "no_oss",  # TODO(b/220688154): Enable OSS tests.
    ],
    deps = [
        ":quantize_model",
        "//tensorflow:tensorflow_py",
    ],
)
