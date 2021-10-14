"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "25a3463c44ee4fccad19b6d8b30899f3d7bd5279"
    LLVM_SHA256 = "b6ba02f741aeaa9f6f93ef4274208cb7ae6bb5f6c04c296c0e3f94f609c59c5b"

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
