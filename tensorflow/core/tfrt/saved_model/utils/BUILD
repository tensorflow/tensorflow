load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
    ],
)

cc_library(
    name = "serialize_bef_utils",
    srcs = ["serialize_bef_utils.cc"],
    hdrs = ["serialize_bef_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
    ],
)

tf_cc_test(
    name = "serialize_bef_utils_test",
    srcs = ["serialize_bef_utils_test.cc"],
    data = [
        "//tensorflow/compiler/mlir/tfrt/tests/saved_model:testdata",
    ],
    deps = [
        ":serialize_bef_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/core/tfrt/utils",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:bef",
    ],
)
