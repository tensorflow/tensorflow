# Description:
#   Tensorflow / MLIR utils.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "string_container_utils",
    hdrs = ["string_container_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "array_container_utils",
    hdrs = ["array_container_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)
