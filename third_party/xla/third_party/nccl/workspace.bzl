"""NCCL (pronounced "Nickel") is a multi-GPU collective communications library that provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter, as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over InfiniBand or TCP/IP sockets across nodes."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nccl_archive",
        build_file = "//third_party/nccl:archive.BUILD",
        patch_file = ["//third_party/nccl:archive.patch"],
        sha256 = "e67239212c395bfdb398a7519491840d06fdf6b599c299f97c7ed0109777bba1",
        strip_prefix = "nccl-2.29.7-1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/nccl/archive/refs/tags/v2.29.7-1.tar.gz"),
    )
