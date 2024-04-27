"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-2a7e9f1256ddc48186c86dff7a00e189b47e5310",
        sha256 = "044d2f5738e677c5f0f1ff9fb616a0245af67d09e42ae3514c73ba50cea0e4a5",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/2a7e9f1256ddc48186c86dff7a00e189b47e5310.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
