# Tools and utilities for analyzing the BufferDebugLogProto dumps.

load("@xla//third_party/rules_python/python:py_test.bzl", "py_test")
load("//xla:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "checksum_mismatch_report",
    srcs = ["checksum_mismatch_report.py"],
    deps = [
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
    ],
)

py_test(
    name = "checksum_mismatch_report_test",
    srcs = ["checksum_mismatch_report_test.py"],
    strict_deps = True,
    deps = [
        ":checksum_mismatch_report",
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_binary(
    name = "check_thunk_output_consistency",
    srcs = [
        "check_thunk_output_consistency.py",
    ],
    main = "check_thunk_output_consistency.py",
    deps = [
        ":checksum_mismatch_report",
        "//xla/backends/gpu/runtime:buffer_debug_log_proto_py",
        "//xla/backends/gpu/runtime:thunk_proto_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@com_google_protobuf//:protobuf_python",
    ],
)
