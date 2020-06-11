load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["logging.h"])

common_copts = ["-Wall"]

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_mac",  # TODO(b/148247402): flatbuffers import broken on Mac OS.
    ],
    deps = [
        ":test_utils",
        ":visualize",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_binary(
    name = "convert_image_to_csv",
    srcs = ["convert_image_to_csv.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
    ],
)

py_test(
    name = "convert_image_to_csv_test",
    srcs = ["convert_image_to_csv_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":convert_image_to_csv",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "strip_strings",
    srcs = ["strip_strings.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":flatbuffer_utils",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "randomize_weights",
    srcs = ["randomize_weights.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":flatbuffer_utils",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_mac",  # TODO(b/148247402): flatbuffers import broken on Mac OS.
    ],
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "@flatbuffers//:runtime_py",
    ],
)

cc_binary(
    name = "generate_op_registrations",
    srcs = ["gen_op_registration_main.cc"],
    deps = [
        ":command_line_flags",
        ":gen_op_registration",
        "//tensorflow/lite:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gen_op_registration",
    srcs = ["gen_op_registration.cc"],
    hdrs = ["gen_op_registration.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "gen_op_registration_test",
    srcs = ["gen_op_registration_test.cc"],
    data = [
        "//tensorflow/lite:testdata/0_subgraphs.bin",
        "//tensorflow/lite:testdata/2_subgraphs.bin",
        "//tensorflow/lite:testdata/empty_model.bin",
        "//tensorflow/lite:testdata/test_model.bin",
        "//tensorflow/lite:testdata/test_model_broken.bin",
        "//tensorflow/lite:testdata/test_model_versioned_ops.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":gen_op_registration",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "verifier_test",
    size = "small",
    srcs = ["verifier_test.cc"],
    data = [
        "//tensorflow/lite:testdata/sparse_tensor.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":verifier",
        "//tensorflow/core:framework_lite",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = common_copts,
)

cc_library(
    name = "tool_params",
    srcs = ["tool_params.cc"],
    hdrs = ["tool_params.h"],
    copts = tflite_copts(),
    deps = [":logging"],
)

cc_test(
    name = "tool_params_test",
    srcs = ["tool_params_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    copts = tflite_copts(),
    deps = ["//tensorflow/lite/tools:logging"],
)

cc_test(
    name = "command_line_flags_test",
    srcs = ["command_line_flags_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":command_line_flags",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_flex_ops",
    srcs = ["list_flex_ops.cc"],
    hdrs = ["list_flex_ops.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:util",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

# This tool list flex ops and kernels inside a TFLite file.
# It is used to generate header file for selective registration.
tf_cc_binary(
    name = "list_flex_ops_main",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_flex_ops",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "list_flex_ops_test",
    srcs = ["list_flex_ops_test.cc"],
    data = [
        "//tensorflow/lite:testdata/0_subgraphs.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
        "//tensorflow/lite:testdata/softplus_flex.bin",
        "//tensorflow/lite:testdata/test_model.bin",
        "//tensorflow/lite:testdata/test_model_broken.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":list_flex_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

py_binary(
    name = "zip_files",
    srcs = ["zip_files.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

tflite_portable_test_suite()
