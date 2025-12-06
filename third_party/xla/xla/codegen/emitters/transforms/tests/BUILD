load("//xla:lit.bzl", "lit_test_suite")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_exec_properties",
    "tf_gpu_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

_GPU_TESTS = [
    "lower_tensors.mlir",
    "vectorize_loads_stores.mlir",
]

lit_test_suite(
    name = "tests",
    srcs = glob(
        include = ["*.mlir"],
        exclude = _GPU_TESTS,
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

lit_test_suite(
    name = "gpu_tests",
    srcs = _GPU_TESTS,
    cfg = "//xla:lit.cfg.py",
    default_tags = tf_gpu_tests_tags(),
    exec_properties = tf_exec_properties({"tags": tf_gpu_tests_tags()}),
    hermetic_cuda_data_dir = "%S/../../../../../../../cuda_nvcc",
    tools = [
        "//xla/codegen/tools:emitters_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
