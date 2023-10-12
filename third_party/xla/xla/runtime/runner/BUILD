load("//xla:strict.default.bzl", "py_strict_library", "py_strict_test")  # maybe @unused in OSS
load("//xla:xla.bzl", "xla_py_proto_library")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "runner_lib",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":runner_proto_cc",
        "//xla/runtime:arguments",
        "//xla/runtime:executable",
        "//xla/runtime:jit_executable",
        "//xla/runtime:logical_result",
        "//xla/runtime:results",
        "//xla/runtime:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/util:command_line_flags",
    ],
)

tf_proto_library(
    name = "runner_proto",
    srcs = ["runner.proto"],
    cc_api_version = 2,
    protodeps = ["//xla:xla_data_proto"],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "runner_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":runner_proto"],
)

xla_py_proto_library(
    name = "xla_data_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = ["//xla:xla_data_proto"],
)

py_strict_library(
    name = "runner",
    testonly = True,
    srcs = ["runner.py"],
    deps = [
        ":runner_proto_py",
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py",
    ],
)

# copybara:uncomment_begin(b/254857628)
# py_strict_test(
#     name = "testlib_runner_test",
#     size = "small",
#     srcs = ["testlib_runner_test.py"],
#     data = [":testlib_runner"],
#     python_version = "PY3",
#     srcs_version = "PY3",
#     deps = [
#         ":runner",
#         "//third_party/py/numpy",
#         "@absl_py//absl/testing:absltest",
#     ],
# )
#
# cc_binary(
#     name = "testlib_runner",
#     testonly = True,
#     srcs = ["testlib_runner.cc"],
#     deps = [
#         ":runner_lib",
#         "//xla/mlir/runtime/transforms/tests:testlib_pipeline",
#     ],
# )
# copybara:uncomment_end
