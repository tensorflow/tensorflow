# Description:
#   Werkzeug provides utilities for making WSGI applications

licenses(["notice"])  # BSD 3-Clause

exports_files(["LICENSE"])

# Note: this library includes test code. Consider creating a testonly target.
py_library(
    name = "werkzeug",
    srcs = glob(["werkzeug/werkzeug/*.py"]),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
