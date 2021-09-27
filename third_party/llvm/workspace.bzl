"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "f7e82e4fa849376ea9226220847a098dc92d74a0"
    LLVM_SHA256 = "b8772661e4888770a2d8fa6bbb71fc5f8186a537f61e7d90a2177baadc1556d8"

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
