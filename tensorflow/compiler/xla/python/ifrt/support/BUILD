load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "sharding_param_to_op_sharding",
    srcs = ["sharding_param_to_op_sharding.cc"],
    hdrs = ["sharding_param_to_op_sharding.h"],
    visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "sharding_param_to_op_sharding_test",
    srcs = ["sharding_param_to_op_sharding_test.cc"],
    deps = [
        ":sharding_param_to_op_sharding",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
