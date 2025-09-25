"""Provides the repository macro to import Rocm-Device-Libs"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Rocm-Device-Libs."""
    LLVM_COMMIT = "c93c6e5451544e9ead12f2d2b15e1969b9a1bd04"
    LLVM_SHA256 = "f715a0a9c3c1a2b09a79939016ed53a0cbd454f7b0ea4ef32878433275c7b16c"

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
