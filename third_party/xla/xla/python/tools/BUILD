# NOTE: We can't use `pytype_pybind_extension` nor `pytype_strict_contrib_test`
# because the OSS versions of these files do not include ports of those rules.
# We must instead use `tsl_pybind_extension` and `py_strict_test`.
load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_test")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files([
    "__init__.py",
    "types.py",
    "_types.pyi",
])

# NOTE: This wrapper library is necessary in order to capture the Python
# dependencies of our extension (namely `ml_dtypes`).  Although the
# underlying `pybind_extension` rule has a `py_deps` argument for capturing
# such dependencies directly, the `tsl_pybind_extension` rule doesn't expose
# that `py_deps` argument for us to use.
#
# NOTE: On the OSS side, the `pytype_strict_library` rule is changed into
# the non-typed rule, which in turn causes an error about the `pytype_srcs`
# field.  The "..:xla_client" target gets around this by adding a custom
# copybara rule; but in lieu of adding yet another custom rule to maintain,
# we just use the generic copybara mechanism for commenting the field out
# on the OSS side.
# TODO(wrengr,phawkins): Once cl/619904840 lands, we can remove the
# pragma and the preceding commentary.
pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    # copybara:uncomment pytype_srcs = ["_types.pyi"],
    srcs_version = "PY3",
    # Cannot build this on OSS because the ":xla_data_proto_py_pb2"
    # dependency isn't part of the public API.
    tags = ["no_oss"],
    visibility = ["//visibility:public"],
    deps = [
        ":_types",  # buildcleaner: keep
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py_pb2",
        "@ml_dtypes",
    ],
)

# NOTE: Copybara detects the `tsl_pybind_extension` rule and automatically
# injects the "@com_google_protobuf//:protobuf_python" python dependency
# required by "@pybind11_protobuf//pybind11_protobuf:native_proto_caster".
tsl_pybind_extension(
    name = "_types",
    srcs = ["_types.cc"],
    pytype_deps = ["//third_party/py/numpy"],
    pytype_srcs = ["_types.pyi"],
    # Users should depend on ":types" instead.
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/nanobind",
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:status_casters",
        "//xla/python:logging",
        "//xla/python:nb_numpy",
        "//xla/python:types",
        "//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    # Cannot build this on OSS because the ":xla_data_proto_py_pb2"
    # dependency isn't part of the public API.
    tags = ["no_oss"],
    deps = [
        ":types",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//xla:xla_data_proto_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
