# Tests for DTensor interfaces.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tf_cc_test(
    name = "dtensor_operation_test",
    srcs = ["dtensor_operation_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/dtensor/cc:dtensor_operation",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "slice_util_test",
    srcs = ["slice_util_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:slice_util",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

tf_cc_test(
    name = "tensor_layout_test",
    srcs = ["tensor_layout_test.cc"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:dstatus",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "layout_to_xla_sharding_test",
    srcs = ["layout_to_xla_sharding_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:dstatus",
        "//tensorflow/dtensor/cc:layout_to_xla_sharding",
        "//tensorflow/dtensor/cc:tensor_layout",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
    ],
)

tf_cc_test(
    name = "executable_manager_test",
    srcs = ["executable_manager_test.cc"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/dtensor/cc:dtensor_device_util",
        "//tensorflow/dtensor/cc:dtensor_operation",
        "//tensorflow/dtensor/cc:tensor_layout",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "spmd_expander_test",
    srcs = ["spmd_expander_test.cc"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/dtensor/cc:dstatus",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/mlir:spmd_expander",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
