load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@llvm-project//mlir:run_lit.sh",
    # TODO(b/177569789): Fix below tests to use V2.
    exclude = [
        "layout_propagation.mlir",
        "spmd_fill.mlir",
        "spmd_metadata.mlir",
        "spmd_reduction.mlir",
        "spmd_tile.mlir",
    ],
    tags_override = {
        "dtensor_embedding_v2.mlir": ["no_oss"],
        "embedding_optimizer.mlir": ["no_oss"],
        "dtensor_embedding_checkpoint.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
        "move_compilation_to_host.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
        "spmd_dtensor_ops.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
        "spmd_embedding.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":dtensor-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

tf_cc_binary(
    name = "dtensor-opt",
    srcs = ["dtensor_mlir_opt_main.cc"],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_dialect_registration",
        "//tensorflow/core:ops",
        "//tensorflow/dtensor/cc:dtensor_ops",
        "//tensorflow/dtensor/mlir:create_dtensor_mlir_passes",
        "//tensorflow/dtensor/mlir:dtensor_mlir_passes",
        "//tensorflow/dtensor/mlir:tf_dtensor_dialect",
        "//tensorflow/dtensor/mlir/dtensor_dialect:Dialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@stablehlo//:register",
    ],
)
