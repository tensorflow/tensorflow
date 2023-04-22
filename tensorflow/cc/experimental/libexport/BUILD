load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility =
        [
            "//tensorflow/cc/experimental/libtf:__subpackages__",
            "//tensorflow/cc/saved_model:__subpackages__",
            "//tensorflow/python/saved_model:__subpackages__",
        ],
    licenses = ["notice"],
)

cc_library(
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    deps = [
        "//tensorflow/core/platform:macros",
    ],
)

cc_library(
    name = "save",
    srcs = ["save.cc"],
    hdrs = ["save.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
    ],
)

tf_cc_test(
    name = "save_test",
    size = "small",
    srcs = ["save_test.cc"],
    deps = [
        ":save",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
    ],
)

cc_library(
    name = "load",
    srcs = ["load.cc"],
    hdrs = ["load.h"],
    deps = [
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_cc_test(
    name = "load_test",
    size = "small",
    srcs = ["load_test.cc"],
    deps = [
        ":load",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
    ],
)

# Linked directly into ":tensorflow_framework".
cc_library(
    name = "metrics_impl",
    srcs = [
        "metrics.cc",
        "metrics.h",
    ],
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "metrics",
    hdrs = ["metrics.h"],
    visibility = [
        "//tensorflow/cc/saved_model:__subpackages__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/saved_model:__subpackages__",
    ],
    deps = if_static([
        ":metrics_impl",
    ]) + ["//tensorflow/core:lib"],
)

tf_cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cc"],
    deps = [
        ":metrics",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)
