# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("@xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
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
        "//third_party/odml/litert/litert/...",
    ],
)

cc_library(
    name = "tf_tfl_flatbuffer_helpers",
    srcs = ["tf_tfl_flatbuffer_helpers.cc"],
    hdrs = ["tf_tfl_flatbuffer_helpers.h"],
    deps = [
        ":slim_model_importer",
        ":stablehlo_tfl_pipeline",
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:model_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
    ],
)

cc_library(
    name = "stablehlo_tfl_pipeline",
    srcs = ["stablehlo_tfl_pipeline.cc"],
    hdrs = ["stablehlo_tfl_pipeline.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite:common",
        "//tensorflow/compiler/mlir/lite:converter_flags_proto_cc",
        "//tensorflow/compiler/mlir/lite:flatbuffer_export",
        "//tensorflow/compiler/mlir/lite:optimize_broadcast_like_pass_options",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/core:macros",
        "//tensorflow/compiler/mlir/lite/debug",
        "//tensorflow/compiler/mlir/lite/stablehlo:drop_shape_assertions",
        "//tensorflow/compiler/mlir/lite/stablehlo:prepare_hlo",
        "//tensorflow/compiler/mlir/lite/stablehlo:tfl_legalize_hlo",
        "//tensorflow/compiler/mlir/lite/stablehlo:unfold_splat_constant_pass",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:vhlo_ops",
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/mlir_hlo:stablehlo_extension_passes",
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
    name = "slim_model_importer",
    srcs = ["slim_model_importer.cc"],
    hdrs = ["slim_model_importer.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/stablehlo:drop_shape_assertions",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/service:hlo_proto_cc",
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

py_library(
    name = "wrap_converter",
    srcs = [
        "wrap_converter.py",
    ],
    strict_deps = True,
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_converter_api",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib_py",
        "//tensorflow/python:pywrap_tensorflow",
    ] + if_pywrap(
        if_true = [] + select({
            "//conditions:default": [],
            # This reqirement is for LiteRT only as tensorflow/compiler/mlirt/lite is
            # duplicated in LiteRT/tflite/converter.
            "//tensorflow:disable_tf_lite_py": [
                "//tensorflow/lite/python:pywrap_tflite",
            ],
        }),
    ),
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
        "//visibility:public",
    ],
    deps = [
        ":flatbuffer_to_mlir",
        ":graphdef_to_tfl_flatbuffer",
        ":jax_to_tfl_flatbuffer",
        ":saved_model_to_tfl_flatbuffer",
        ":slim_model_importer",
        ":tf_tfl_flatbuffer_helpers",
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
        "@xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@xla//xla/tsl/platform:status",
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
    visibility = ["//visibility:public"],
    deps = [
        ":converter_python_api",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
        "@xla//third_party/python_runtime:headers",
    ],
)
