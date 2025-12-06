"""Provides the repository macro to import Rocm-Device-Libs"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Rocm-Device-Libs."""
    LLVM_COMMIT = "bc1578256b4894680e0d1c2552cabd2d93803c37"
    LLVM_SHA256 = "dfef281553a980a960ac38c697682f2f7c8eebafccfd32ce8e481eb290482acd"

    tf_http_archive(
        name = "rocm_device_libs",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}/amd/device-libs".format(commit = LLVM_COMMIT),
        urls = tf_mirror_urls("https://github.com/ROCm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)),
        build_file = "//third_party/rocm_device_libs:rocm_device_libs.BUILD",
        patch_file = [
            "//third_party/rocm_device_libs:prepare_builtins.patch",
        ],
        link_files = {
            "//third_party/rocm_device_libs:build_defs.bzl": "build_defs.bzl",
        },
    )
