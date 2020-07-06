# Library for using LLVM style RTTI in TensorFlow.
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "llvm_rtti",
    hdrs = ["llvm_rtti.h"],
    deps = [
        "@llvm-project//llvm:Support",
    ],
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "llvm_rtti.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
