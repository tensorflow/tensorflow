load("//tensorflow:tensorflow.bzl", "pytype_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pytype_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

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
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:ArithmeticDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op_and_kernels",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_freeze_variables",
        "//tensorflow/compiler/mlir/tensorflow:tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:statusor",
    ] + if_static(
        extra_deps = [
            "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        ],
        otherwise = [
            "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc_headers_only",
        ],
    ),
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
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:calibrator_singleton",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:custom_aggregator_op_and_kernels",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
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
        "@com_google_absl//absl/strings",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/python/lib/core:pybind11_lib",
    ] + if_static(
        extra_deps = [
            "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        ],
        otherwise = [
            "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc_headers_only",
        ],
    ),
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
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:tag_constants",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
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
        ":quantize_model_test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
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
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/types",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/types",
    ],
)

tf_py_test(
    name = "representative_dataset_test",
    srcs = ["representative_dataset_test.py"],
    tags = ["no_pip"],  # b/241528672
    deps = [
        ":representative_dataset",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
        "//third_party/py/numpy",
    ],
)
