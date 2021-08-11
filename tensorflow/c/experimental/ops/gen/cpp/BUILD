load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "cpp",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = ["//tensorflow/c/experimental/ops/gen:__pkg__"],
    deps = [
        "//tensorflow/c/experimental/ops/gen/common",
        "//tensorflow/c/experimental/ops/gen/cpp/renderers",
        "//tensorflow/c/experimental/ops/gen/model",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:str_util",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cpp_generator_test",
    size = "small",
    srcs = ["cpp_generator_test.cc"],
    data = ["//tensorflow/c/experimental/ops/gen/cpp/golden"],
    deps = [
        ":cpp",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
