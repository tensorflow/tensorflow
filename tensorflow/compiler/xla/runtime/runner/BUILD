load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "runner_lib",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    deps = [
        ":runner_proto_cc",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/compiler/xla/runtime:logical_result",
        "//tensorflow/compiler/xla/runtime:results",
        "//tensorflow/compiler/xla/runtime:types",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_proto_library(
    name = "runner_proto",
    srcs = ["runner.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

xla_py_proto_library(
    name = "runner_pb2",
    api_version = 2,
    deps = [":runner_proto"],
)

xla_py_proto_library(
    name = "xla_data_pb2",
    api_version = 2,
    deps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

py_library(
    name = "runner",
    testonly = True,
    srcs = ["runner.py"],
    deps = [
        ":runner_proto_py",
        "//tensorflow/compiler/xla/python:xla_client",
    ],
)

# copybara:uncomment_begin(b/254857628)
# py_test(
#     name = "testlib_runner_test",
#     size = "small",
#     srcs = ["testlib_runner_test.py"],
#     data = [":testlib_runner"],
#     python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#         ":runner",
#         "@absl_py//absl/testing:absltest",
#         "//third_party/py/numpy",
#     ],
# )
#
# cc_binary(
#     name = "testlib_runner",
#     testonly = True,
#     srcs = ["testlib_runner.cc"],
#     deps = [
#         ":runner_lib",
#         "//tensorflow/compiler/xla/mlir/runtime/transforms/tests:testlib_pipeline",
#     ],
# )
# copybara:uncomment_end
