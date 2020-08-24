# Description:
#   Backports for the typing module to older Python versions. See
#   https://github.com/python/typing/blob/master/typing_extensions/README.rst

licenses(["notice"])  # PSF

exports_files(["LICENSE"])

py_library(
    name = "typing_extensions",
    srcs = ["src_py3/typing_extensions.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
