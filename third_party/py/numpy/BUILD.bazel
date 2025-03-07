licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

alias(
    name = "numpy",
    actual = "@pypi_numpy//:pkg",
)

alias(
    name = "headers",
    actual = "@pypi_numpy//:numpy_headers",
)

genrule(
    name = "dummy",
    outs = ["tf_numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
