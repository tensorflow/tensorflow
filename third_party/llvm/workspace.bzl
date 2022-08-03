"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "955cc56af448301e2d353df90ca1f0c6a74118f2"
    LLVM_SHA256 = "e4566371654993c5520e9c31e0bb468e2baa629747857a41976c6047aa37ceaf"

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
            "//third_party/llvm:infer_type.patch",  # TODO(b/231285230): remove once resolved
            "//third_party/llvm:build.patch",
            "//third_party/llvm:macos_build_fix.patch",
            "//third_party/llvm:temp_fix_for_llvm_integrate.patch",
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
