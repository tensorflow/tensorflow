"""Loads NVIDIA NVTX 3 headers."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvtx_archive",
        build_file = "//third_party:nvtx/BUILD.bazel",
        sha256 = "f244c5eec33f9769123a755dda1e9b80339345ab278cf9542ff34677c88804b5",
        strip_prefix = "NVTX-3.5.0/c/include",
        urls = tf_mirror_urls(
            "https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.5.0.tar.gz",
        ),
    )
