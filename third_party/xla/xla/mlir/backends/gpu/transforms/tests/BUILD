load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            "add_concurrent_regions.mlir",
            "add_hlo_trace.mlir",
            "gpu_launch.mlir",
            "gpu_memcpy.mlir",
            "gpu_memset.mlir",
            "lmhlo_case.mlir",
            "lmhlo_custom_call.mlir",
            "lmhlo_fft.mlir",
            "lmhlo_gpu_cholesky.mlir",
            "lmhlo_gpu_conv.mlir",
            "lmhlo_gpu_cublas_lt_matmul.mlir",
            "lmhlo_gpu_gemm.mlir",
            "lmhlo_infeed.mlir",
            "lmhlo_outfeed.mlir",
            "lmhlo_send_recv.mlir",
            "lmhlo_while.mlir",
            "memref_get_global_to_arg.mlir",
            "outline_cuda_graphs.mlir",
            "stream_assignment.mlir",
        ],
        include = ["*.mlir"],
    ),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/mlir/backends/gpu:xla-gpu-opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
