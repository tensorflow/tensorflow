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

lit_test_suite(
    name = "tests",
    srcs = glob(["**/*.hlo"]),
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_cuda_tests_tags(),
    exec_properties = tf_exec_properties({"tags": tf_cuda_tests_tags()}),
    hermetic_cuda_data_dir = "%S/../../../../../../../cuda_nvcc",
    tools = [
        "//xla/codegen/tools:emitters_opt",
        "//xla/service/gpu/fusions/tools:fusion_to_mlir",
        "//xla/service/gpu/fusions/tools:test_correctness",
        "@llvm-project//llvm:FileCheck",
    ],
)
