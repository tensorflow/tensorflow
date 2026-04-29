"""Loads the nvtx library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvtx_archive",
        build_file = "@xla//third_party:nvtx/BUILD.bazel",
        sha256 = "5a581c3234c5a6b2fd94363e3fdd5a4f5d2a3d9c53c4b9442b0784e6cdfe722c",
        strip_prefix = "NVTX-2942f167cc30c5e3a44a2aecd5b0d9c07ff61a07/c/include",
        urls = tf_mirror_urls("https://github.com/NVIDIA/NVTX/archive/2942f167cc30c5e3a44a2aecd5b0d9c07ff61a07.tar.gz"),
    )
