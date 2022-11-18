"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "0cf888514454350cd97ab79cdb4a73e7f189eea0"
    LLVM_SHA256 = "515bd27c1e4dda74603237a84e89b1c31b96aa91934fefb196b98bf18a443991"

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
