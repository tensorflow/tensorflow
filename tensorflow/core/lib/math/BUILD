package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: math_util_test.

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

exports_files([
    "math_util.h",
    "math_util_test.cc",
])
