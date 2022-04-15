package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "views",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = ["//tensorflow/c/experimental/ops/gen/cpp/renderers:__pkg__"],
    deps = [
        "//tensorflow/c/experimental/ops/gen/common",
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
