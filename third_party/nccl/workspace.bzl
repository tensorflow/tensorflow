"""Loads the nccl library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nccl_archive",
        build_file = "@xla//third_party/nccl:archive.BUILD",
        patch_file = ["@xla//third_party/nccl:archive.patch"],
        sha256 = "98e6262bd55932c51e7c8ffc50cc764f019e4b94a8fd6694d839ae828ec8d128",
        strip_prefix = "nccl-2.27.7-1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/nccl/archive/refs/tags/v2.27.7-1.tar.gz"),
    )
