load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite", "verifier_internal_visibility_allowlist")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files([
    "logging.h",
])

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "visualize_lib",
    srcs = ["visualize.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:session",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "convert_image_to_csv_test",
    srcs = ["convert_image_to_csv_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "reverse_xxd_dump_from_cc",
    srcs = ["reverse_xxd_dump_from_cc.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "randomize_weights",
    srcs = ["randomize_weights.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/lite/python:schema_py",
        "//tensorflow/python/platform",
        "@flatbuffers//:runtime_py",
    ],
)

py_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_utils",
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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "verifier",
    hdrs = ["verifier.h"],
    deps = [
        ":verifier_internal",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/api:op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/tools:verifier",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "verifier_internal",
    hdrs = ["verifier_internal.h"],
    visibility = verifier_internal_visibility_allowlist() + [
        "//tensorflow/lite/core/shims:__subpackages__",
        "//tensorflow/lite/java/src/main/native:__pkg__",
    ],
    deps = [
        "//tensorflow/lite/core/tools:verifier_internal",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = tflite_copts_warnings(),
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
    deps = [
        "//tensorflow/lite/tools:logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "command_line_flags_test",
    srcs = ["command_line_flags_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":command_line_flags",
        ":tool_params",
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
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_utils",
        "@flatbuffers",
        "@jsoncpp_git//:jsoncpp",
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

cc_library(
    name = "list_flex_ops_main_lib",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_flex_ops",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "list_flex_ops_no_kernel",
    srcs = ["list_flex_ops_no_kernel.cc"],
    hdrs = ["list_flex_ops.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/schema:schema_utils",
        "@jsoncpp_git//:jsoncpp",
    ],
)

tf_cc_binary(
    name = "list_flex_ops_no_kernel_main",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":list_flex_ops_no_kernel",
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
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
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

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":logging",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "model_loader",
    srcs = ["model_loader.cc"],
    hdrs = ["model_loader.h"],
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:model_builder",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "model_loader_test",
    srcs = ["model_loader_test.cc"],
    data = ["//tensorflow/lite/java/demo/app/src/main/assets:mobilenet_v1_1.0_224.tflite"],
    deps = [
        ":model_loader",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

tflite_portable_test_suite()
