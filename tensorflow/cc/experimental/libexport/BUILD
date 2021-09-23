load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
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
