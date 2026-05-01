"""Module extension for llvm."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

def _llvm_extension_impl(mctx):  # @unused
    llvm_configure(
        name = "llvm-project",
        targets = [
            "AArch64",
            "AMDGPU",
            "ARM",
            "NVPTX",
            "PowerPC",
            "RISCV",
            "SystemZ",
            "X86",
            "SPIRV",
        ],
    )

llvm_extension = module_extension(
    implementation = _llvm_extension_impl,
)
