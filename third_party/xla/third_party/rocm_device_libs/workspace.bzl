"""Provides the repository macro to import Rocm-Device-Libs"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Rocm-Device-Libs."""
    LLVM_COMMIT = "fcc50fb091b7c75d8f6c9a6554d0b004bc0cd474"
    LLVM_SHA256 = "fa9089d3134bd32d2b05a141006b9261e441c1d80b75782db0dcb154b6a60561"

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
