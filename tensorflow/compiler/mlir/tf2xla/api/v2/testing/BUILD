load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tf2xla/api/v2:__subpackages__",
    ],
)

cc_library(
    name = "compile_mlir",
    testonly = True,
    srcs = ["compile_mlir.cc"],
    hdrs = ["compile_mlir.h"],
    deps = [
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:legalize_tf",
        "//tensorflow/compiler/mlir/tf2xla/internal/utils:test_metadata_config",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

cc_library(
    name = "utils",
    testonly = True,
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = ["//tensorflow/core/platform:resource_loader"],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
    ],
)
