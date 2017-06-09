# Description:
# Import of html5lib library.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-like notice-style license, see LICENSE file

exports_files(["LICENSE"])

py_library(
    name = "org_html5lib",
    srcs = glob(["html5lib/**/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)
