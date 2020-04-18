load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

glob_lit_tests(
    data = [":test_utilities"],
    # TODO(141337082): Figure out why the tests do not run successfully in open
    # source.
    default_tags = ["no_oss"] + tf_cuda_tests_tags(),
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [
        # TODO(b/149302060) Reenable once fusion is fixed.
        "iota_add_multiply.hlo",
        # TODO(b/137624192): Reenable once we can fuse reductions.
        "fused_reduce.hlo",
    ],
    test_file_exts = ["hlo"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/xla/service/mlir_gpu:xla-gpu-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
