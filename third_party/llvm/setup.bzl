"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

def llvm_setup(name):
    # Disable terminfo and zlib that are bundled with LLVM.
    llvm_disable_optional_support_deps()

    # Build @llvm-project from @llvm-raw using overlays.
    llvm_configure(name = name)
