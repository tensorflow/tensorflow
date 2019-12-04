# Description:
#   BUILD file for the Python bindings.

licenses(["notice"])  # Apache 2.0

# Export the BUILD file so automated tooling can check licenses
exports_files(["BUILD"])

package(
    default_visibility = ["@local_config_mlir//:friends"],
)

#
# Pybind route uses exceptions and py_extension.
#
py_extension(
    name = "_pybind",
    srcs = ["pybind.cpp"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "pybind",
    deps = [
        "//third_party/llvm/llvm:ir",
        "//third_party/llvm/llvm:support",
        "//third_party/pybind11",
        "@local_config_mlir//:AffineToStandardTransforms",
        "@local_config_mlir//:EDSC",
        "@local_config_mlir//:EDSCInterface",
        "@local_config_mlir//:ExecutionEngine",
        "@local_config_mlir//:ExecutionEngineUtils",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:LLVMTransforms",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:TargetLLVMIR",
        "@local_config_mlir//:Transforms",
    ],
)
