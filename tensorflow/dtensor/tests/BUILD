# Tests for DTensor interfaces.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tf_cc_test(
    name = "tensor_layout_test",
    srcs = ["tensor_layout_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "//tensorflow/tsl/platform:status_matchers",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "layout_to_xla_sharding_test",
    srcs = ["layout_to_xla_sharding_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:layout_to_xla_sharding",
        "//tensorflow/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "executable_manager_test",
    srcs = ["executable_manager_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:dtensor_device_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)
