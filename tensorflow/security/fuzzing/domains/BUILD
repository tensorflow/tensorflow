# Fuzzing input domains.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

cc_library(
    name = "function_def",
    testonly = 1,
    hdrs = ["function_def.h"],
    visibility = [
        "//tensorflow/security/fuzzing/core/framework:__subpackages__",
        "//tensorflow/security/fuzzing/core/function:__subpackages__",
    ],
    deps = [
        "//tensorflow/core/framework:function_fuzz",
    ],
)