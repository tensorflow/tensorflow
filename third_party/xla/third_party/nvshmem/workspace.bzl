"""NVSHMEM - NVIDIA Shared Memory"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nvshmem",
        strip_prefix = "nvshmem_src",
        sha256 = "2146ff231d9aadd2b11f324c142582f89e3804775877735dc507b4dfd70c788b",
        urls = tf_mirror_urls("https://developer.download.nvidia.com/compute/redist/nvshmem/3.1.7/source/nvshmem_src_3.1.7-1.txz"),
        build_file = "//third_party/nvshmem:nvshmem.BUILD",
        type = "tar",
    )
