load("//tensorflow:tensorflow.bzl", "py_binary", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite:__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
)

filegroup(
    name = "toco_python_api_hdrs",
    srcs = [
        "toco_python_api.h",
    ],
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
)

cc_library(
    name = "toco_python_api",
    srcs = ["toco_python_api.cc"],
    hdrs = ["toco_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        "@com_google_protobuf//:protobuf_headers",
        "//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/c:kernels",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/python/interpreter_wrapper:python_utils",
        "//tensorflow/lite/python/interpreter_wrapper:python_error_reporter",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/toco/logging:conversion_log_util",
        "//tensorflow/lite/toco:model_flags_proto_cc",
        "//tensorflow/lite/toco:toco_convert",
        "//tensorflow/lite/toco/logging:toco_conversion_log_proto_cc",
        "//tensorflow/lite/toco:toco_flags_proto_cc",
        "//tensorflow/lite/toco:toco_graphviz_dump_options",
        "//tensorflow/lite/toco:toco_port",
        "//tensorflow/lite/toco:toco_tooling",
        "//tensorflow/lite/toco:tooling_util",
        "//tensorflow/lite/toco:types_proto_cc",
        "//tensorflow/compiler/mlir/lite/python:flatbuffer_to_mlir",
        "//tensorflow/compiler/mlir/lite/python:graphdef_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite/python:saved_model_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite/python:jax_to_tfl_flatbuffer",
        "//tensorflow/compiler/mlir/lite/quantization/lite:quantize_model",
        "//tensorflow/compiler/mlir/lite/sparsity:sparsify_model",
        "//tensorflow/compiler/mlir/lite/metrics:error_collector",
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

# Compatibility stub. Remove when internal customers moved.
py_library(
    name = "tensorflow_wrap_toco",
    srcs = ["tensorflow_wrap_toco.py"],
    srcs_version = "PY3",
    visibility = [
        "//learning/expander/pod/deep_pod/utils:__subpackages__",
        "//research/handwriting/converters/tflite:__subpackages__",
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/python:_pywrap_toco_api",
    ],
)

py_binary(
    name = "toco_from_protos",
    srcs = ["toco_from_protos.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:_pywrap_toco_api",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "@absl_py//absl:app",
    ],
)

tf_py_test(
    name = "toco_from_protos_test",
    srcs = ["toco_from_protos_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/toco:model_flags_proto_py",
        "//tensorflow/lite/toco:toco_flags_proto_py",
    ],
)
