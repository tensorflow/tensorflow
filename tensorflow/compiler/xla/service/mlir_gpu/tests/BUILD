load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
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
    default_tags = tf_cuda_tests_tags() + [
        "no_pip",
        "config-cuda-only",
        "no_rocm",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [
        # TODO(b/137624192): Reenable once we can fuse reductions.
        "fused_reduce.hlo",
    ],
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
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
