load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//xla:lit.bzl", "lit_test_suite")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

copy_file(
    name = "cpu_fusion_to_mlir_copy",
    testonly = True,
    src = "//xla/backends/cpu/codegen/tools:fusion_to_mlir",
    out = "cpu_fusion_to_mlir",
    is_executable = True,
)

copy_file(
    name = "gpu_fusion_to_mlir_copy",
    testonly = True,
    src = "//xla/backends/gpu/codegen/tools:fusion_to_mlir",
    out = "gpu_fusion_to_mlir",
    is_executable = True,
)

lit_test_suite(
    name = "tests",
    srcs = glob(["**/*.hlo"]),
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_cuda_tests_tags(),
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
    hermetic_cuda_data_dir = "%S/../../../../../../cuda_nvcc",
    tools = [
        ":cpu_fusion_to_mlir_copy",
        ":gpu_fusion_to_mlir_copy",
        "//xla/backends/cpu/codegen/tools:cpu_test_correctness",
        "//xla/backends/gpu/codegen/tools:gpu_test_correctness",
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
