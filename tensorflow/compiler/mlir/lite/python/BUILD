load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        "//tensorflow:__pkg__",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/mobile/lite/tooling/model_analyzer/...",
        "//learning/brain/mobile/lite/tools/analyzer/...",
        "//tensorflow/lite/python/...",
        "//tensorflow/lite/python/converter/...",
        "//tensorflow/lite/toco/...",
    ],
)

cc_library(
    name = "tf_tfl_flatbuffer_helpers",
    srcs = ["tf_tfl_flatbuffer_helpers.cc"],
    hdrs = ["tf_tfl_flatbuffer_helpers.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "graphdef_to_tfl_flatbuffer",
    srcs = ["graphdef_to_tfl_flatbuffer.cc"],
    hdrs = [
        "graphdef_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "saved_model_to_tfl_flatbuffer",
    srcs = ["saved_model_to_tfl_flatbuffer.cc"],
    hdrs = [
        "saved_model_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "jax_to_tfl_flatbuffer",
    srcs = ["jax_to_tfl_flatbuffer.cc"],
    hdrs = [
        "jax_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/hlo/translate:stablehlo",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

# Smaller version of flatbuffer_translate which only converts flatbuffer to MLIR.
cc_library(
    name = "flatbuffer_to_mlir",
    srcs = [
        "flatbuffer_to_mlir.cc",
    ],
    hdrs = [
        "flatbuffer_to_mlir.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_import",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

py_strict_library(
    name = "wrap_converter",
    srcs = [
        "wrap_converter.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_converter_api",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = ["//visibility:private"],
)

filegroup(
    name = "converter_python_api_hdrs",
    srcs = [
        "converter_python_api.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "converter_python_api",
    srcs = ["converter_python_api.cc"],
    hdrs = ["converter_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        ":flatbuffer_to_mlir",
        ":graphdef_to_tfl_flatbuffer",
        ":jax_to_tfl_flatbuffer",
        ":saved_model_to_tfl_flatbuffer",
        "//tensorflow/c:kernels",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/core:absl_error_model_builder",
        "//tensorflow/compiler/mlir/lite/debug:debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector",
        "//tensorflow/compiler/mlir/lite/python/interpreter_wrapper:python_error_reporter",
        "//tensorflow/compiler/mlir/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/compiler/mlir/lite/quantization/lite:quantize_model",
        "//tensorflow/compiler/mlir/lite/schema:schema_fbs",
        "//tensorflow/compiler/mlir/lite/sparsity:sparsify_model",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@local_xla//xla/tsl/platform:status",
    ] + select({
        # This is required when running `tflite_convert` from `bazel`.
        # It requires to link with TensorFlow Ops to get the op definitions.
        ":tflite_convert_with_select_tf_ops": [
            "//tensorflow/core:ops",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

tf_python_pybind_extension(
    name = "_pywrap_converter_api",
    srcs = [
        "converter_python_api_wrapper.cc",
    ],
    hdrs = [":converter_python_api_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_converter_api.pyi",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":converter_python_api"]),
)
