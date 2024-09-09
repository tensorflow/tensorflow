load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//xla:lit.bzl", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tags_override = {
        "convert_float_nvidia.mlir": [
            # This is a test for a Nvidia specific pass.
            "no_rocm",
        ] + tf_cuda_tests_tags(),
    },
    tools = [
        "//xla/service/gpu/fusions/tools:mlir_fusions_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
