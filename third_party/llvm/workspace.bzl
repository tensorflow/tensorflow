"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "161755770a44faaedc1a5e74a22b91f4d6ef9669"
    LLVM_SHA256 = "87e71f95cfd8aa54bcc54f1b4bb0bd57705766ab3439908671b24b39ab784f75"

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
