load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":__subpackages__",
        ":tf2xla_users",
    ],
)

# Please reach out to tf-bridge-team@ before using the TF2XLA bridge.
package_group(name = "tf2xla_users")

cc_library(
    name = "legalize_tf",
    srcs = ["legalize_tf.cc"],
    hdrs = ["legalize_tf.h"],
    deps = [
        ":device_type_proto_cc",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "legalize_tf_test",
    srcs = ["legalize_tf_test.cc"],
    deps = [
        ":device_type_proto_cc",
        ":legalize_tf",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "device_type_proto",
    srcs = ["device_type.proto"],
    cc_api_version = 2,
)
