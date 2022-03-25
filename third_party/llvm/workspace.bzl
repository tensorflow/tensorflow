"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "d3e5b6f7539b86995aef6e2075c1edb3059385ce"
    LLVM_SHA256 = "e3809cd36cbfe34d78df965be1bf1780f0db3b6f45aee1b4f092c7c4f5a6a644"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = ["//third_party/llvm:macos_build_fix.patch", "//third_party/llvm:getFunctionType.patch"],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
