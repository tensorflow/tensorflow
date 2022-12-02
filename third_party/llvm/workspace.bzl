"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "61aed52c9ec0bca9979922ed2762785ec1f39755"
    LLVM_SHA256 = "104552d387acc5ca1f9cbece26d98e57dd8d3b7346e869d6cf7d0d63d94a468a"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
            "//third_party/llvm:build.patch",
            "//third_party/llvm:mathextras.patch",
            "//third_party/llvm:toolchains.patch",
            "//third_party/llvm:temporary.patch",  # Cherry-picks and temporary reverts. Do not remove even if temporary.patch is empty.
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
