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
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
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
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/tools/optimize:reduced_precision_support",
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
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
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
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:types_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/service:hlo_parser",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
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
    srcs_version = "PY3",
    deps = [
        ":_pywrap_converter_api",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

filegroup(
    name = "converter_python_api_hdrs",
    srcs = [
        "converter_python_api.h",
    ],
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
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
        "//tensorflow/compiler/mlir/lite/debug:debug_options_proto_cc",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector",
        "//tensorflow/compiler/mlir/lite/quantization/lite:quantize_model",
        "//tensorflow/compiler/mlir/lite/sparsity:sparsify_model",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_convert",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:toco_graphviz_dump_options",
        "//tensorflow/lite/toco:toco_port",
        "//tensorflow/lite/toco:toco_tooling",
        "//tensorflow/lite/toco:tooling_util",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/lite/toco/logging:conversion_log_util",
        "//tensorflow/lite/toco/logging:toco_conversion_log_proto_cc",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform:status",
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
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
