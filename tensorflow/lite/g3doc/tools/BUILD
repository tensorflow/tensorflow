load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")
load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "build_py_api_docs",
    srcs = ["build_py_api_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
