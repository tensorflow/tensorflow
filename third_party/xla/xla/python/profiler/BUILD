load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@local_tsl//tsl/profiler/protobuf:xplane_py_pb2",
    ],
)

py_strict_test(
    name = "xplane_visitor_test",
    srcs = ["xplane_visitor_test.py"],
    python_version = "PY3",
    deps = [
        ":xplane_visitor",
        "@absl_py//absl/testing:absltest",
        "@local_tsl//tsl/profiler/protobuf:xplane_py_pb2",
    ],
)
