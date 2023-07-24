load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "attribute",
    srcs = ["attribute.cc"],
    hdrs = ["attribute.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "attribute_test",
    srcs = ["attribute_test.cc"],
    deps = [
        ":attribute",
        "//tensorflow/compiler/mlir/tensorflow",  # build_cleaner: keep
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)
