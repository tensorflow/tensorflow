# Description:
#   Backports for the typing module to older Python versions. See
#   https://github.com/python/typing/blob/master/typing_extensions/README.rst

licenses(["notice"])  # PSF

py_library(
    name = "typing_extensions",
    srcs = ["typing_extensions.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

genrule(
    name = "license",
    srcs = ["@astunparse_license"],
    outs = ["LICENSE"],
    cmd = "cp $< $@",
    visibility = ["//visibility:public"],
)
