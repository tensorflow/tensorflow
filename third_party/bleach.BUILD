# Description:
#   Build file for Bleach.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "org_mozilla_bleach",
    srcs = [
        "bleach/__init__.py",
        "bleach/callbacks.py",
        "bleach/encoding.py",
        "bleach/sanitizer.py",
        "bleach/version.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["@org_html5lib"],
)
