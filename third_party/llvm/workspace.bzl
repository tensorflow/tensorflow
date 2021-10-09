"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "778bf73d7ba693f4bb74d137a70870bce128a19f"
    LLVM_SHA256 = "3a948a5763044a8fbe4b028668f3e83e60c42a914098ec55951a85d5fda94838"

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
