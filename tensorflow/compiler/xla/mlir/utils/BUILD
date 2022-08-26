load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "to_string",
    hdrs = ["to_string.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = ["@llvm-project//llvm:Support"],
)
