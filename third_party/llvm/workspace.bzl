"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "fad14a17a490b1825b0c7c40ace7e83c91af4b8a"
    LLVM_SHA256 = "4e0cea18b548cd1e3d333ad0b22e9ba9bfa33b946b8c6c5a975ad685da154717"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:BUILD.bazel",
        patch_file = "//third_party/llvm:macos_build_fix.patch",
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
