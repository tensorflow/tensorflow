"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "7d659c6ac741f66927938d1e5c6b85deeb294882"
    LLVM_SHA256 = "94225c01f49e4dbe2be24ae78eed663d622ab47770226b7bcec6470d59658fa2"

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
