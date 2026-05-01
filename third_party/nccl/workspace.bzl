"""Loads the nccl library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nccl_archive",
        build_file = "@xla//third_party/nccl:archive.BUILD",
        patch_file = ["@xla//third_party/nccl:archive.patch"],
        sha256 = "e67239212c395bfdb398a7519491840d06fdf6b599c299f97c7ed0109777bba1",
        strip_prefix = "nccl-2.29.7-1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/nccl/archive/refs/tags/v2.29.7-1.tar.gz"),
    )
