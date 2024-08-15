load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    size_override = {
        "decompose_resource_ops.mlir": "medium",
        "layout_optimization_move_transposes_end.mlir": "medium",
        "layout_optimization_to_nhwc.mlir": "medium",
    },
    tags_override = {
        "optimize.mlir": ["no_rocm"],
        "tf_optimize.mlir": ["no_rocm"],
        "tf-reduce-identity.mlir": ["no_windows"],
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":reducer_tester",
        "//tensorflow/compiler/mlir:tf-opt",
        "//tensorflow/compiler/mlir:tf-reduce",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

filegroup(
    name = "reducer_tester",
    testonly = True,
    srcs = glob(
        [
            "reducer/*.sh",
        ],
    ),
)

tf_cc_test(
    name = "xla_sharding_util_test",
    srcs = ["xla_sharding_util_test.cc"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:xla_sharding_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)
