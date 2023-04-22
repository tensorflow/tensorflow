load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_tests",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "common",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    deps = [
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "all_tests",
    size = "small",
    srcs = glob(["*_test.cc"]),
    deps = [
        ":common",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
