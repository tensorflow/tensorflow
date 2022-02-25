"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "e7fb1c15cb85d748c1c4fdd5a2eb5613ec7bef1d"
    LLVM_SHA256 = "2a2ded666d2e1fbda0e5c852b2c4e4a4c290d9694d412a742d28cfde5ebf4351"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = ["//third_party/llvm:macos_build_fix.patch"],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
