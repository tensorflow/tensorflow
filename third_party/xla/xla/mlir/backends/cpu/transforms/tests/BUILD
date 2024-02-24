load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "collective_ops.mlir",
            "fft.mlir",
            "legalize_i1_vector_transfers.mlir",
            "library_ops_to_cpu_runtime.mlir",
            "lmhlo_custom_call.mlir",
            "remove_copies_to_out_params.mlir",
            "rng_bit_generator.mlir",
            "xla_abi_legalization.mlir",
            "xla_cpu_infeed.mlir",
            "xla_cpu_memref_element_cast_to_llvm.mlir",
            "xla_cpu_outfeed.mlir",
        ],
        include = [
            "*.mlir",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/backends/cpu:xla-cpu-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
