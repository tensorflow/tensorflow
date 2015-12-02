licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        "numpy_include/**/*.h",
    ]),
    data = ["//util/python:python_checked"],
    includes = [
        "numpy_include",
    ],
)
