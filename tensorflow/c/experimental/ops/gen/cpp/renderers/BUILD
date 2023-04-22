load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_tests",
)

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "renderers",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = ["//tensorflow/c/experimental/ops/gen/cpp:__pkg__"],
    deps = [
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/cpp/views",
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
    name = "renderer_tests",
    size = "small",
    srcs = glob(
        ["*_test.cc"],
    ),
    deps = [
        ":renderers",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
