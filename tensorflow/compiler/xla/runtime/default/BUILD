load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//tensorflow/compiler/xla/runtime:__pkg__"],
    licenses = ["notice"],
)

cc_library(
    name = "async_values_cache",
    hdrs = ["async_values_cache.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "memory_mapper",
    hdrs = ["memory_mapper.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)
