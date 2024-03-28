load("@local_tsl//tsl:tsl.default.bzl", "tsl_pybind_extension")

# NOTE: We can't use `pytype_pybind_extension` nor `pytype_strict_contrib_test`
# because the OSS versions of these files do not include ports of those rules.
# We must instead use `tsl_pybind_extension` and `py_strict_test`.
load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files([
    "__init__.py",
    "types.py",
    "_types.pyi",
])

# FIXME(wrengr): On the OSS side we're getting """ERROR: package contains
# errors: xla/python/tools: //xla/python/tools:types: no such attribute
# 'pytype_srcs' in 'py_{}library' rule""".  (I added the "{}" to
# disable an erroneous error from "Presubmit:Contents".)
# However, I'm just following the same pattern as in ../BUILD:
# * the `pytype_strict_library` rule for ":xla_client" uses `pytype_srcs`
# * the `tsl_pybind_extension` rule for ":xla_extension" uses `pytype_srcs`

# NOTE: This wrapper library is necessary in order to capture the Python
# dependencies of our extension (namely `ml_dtypes`).  Although the
# underlying `pybind_extension` rule has a `py_deps` argument for capturing
# such dependencies directly, the `tsl_pybind_extension` rule doesn't expose
# that `py_deps` argument for us to use.
pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    # FIXME(wrengr): Commenting this out for now, to see if that's enough
    # to get OSS to build.  See the longer fixme note above.
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
        "//xla/python:logging",
        "//xla/python:types",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/python/lib/core:numpy",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
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
    # TODO(wrengr): Need we add `xla_py_{}test_deps()` to the deps for OSS?
    # The "{}" there is to disable an erroneous error from "Presubmit:Contents"
)
