load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

package(licenses = ["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

tfrt_cc_test(
    name = "tf_jitrt_benchmark_test",
    srcs = ["tf_jitrt_benchmark_test.cc"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_pipeline",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationTransforms",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/jitrt",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
    ],
)
