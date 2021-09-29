"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "2240deb9766cc080b351016b0d7f975d7249b113"
    LLVM_SHA256 = "498ab90611704cba3ee8b084b11bd055882f7c139cf5c9391fdfaecba2191eef"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:BUILD.bazel",
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
