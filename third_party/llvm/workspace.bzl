"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "4f60a42878b0d46bc2cc84d8f0d316cac2c60c9d"
    LLVM_SHA256 = "5cb31c75824a9f55904c3317a69cc9652bf2a2ea85ee71b390eefa0980f9add5"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = "//third_party/llvm:macos_build_fix.patch",
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
