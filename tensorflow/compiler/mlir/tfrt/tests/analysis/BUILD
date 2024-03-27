load("//tensorflow:tensorflow.bzl", "if_oss", "tf_cc_test")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    exclude = ["testdata/**"],
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

tf_cc_test(
    name = "update_op_cost_in_tfrt_mlir_test",
    srcs = ["update_op_cost_in_tfrt_mlir_test.cc"],
    data = [
        "testdata/test.mlir",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:init_tfrt_dialects",
    ],
)
