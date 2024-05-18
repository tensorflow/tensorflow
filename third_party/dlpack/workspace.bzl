"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-bbd2f4d32427e548797929af08cfe2a9cbb3cf12",
        sha256 = "f5dcb30f8a3d1a41d48b5d5b3fe1631dfce11de76f9e7e5bd22c6dc75a758885",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/bbd2f4d32427e548797929af08cfe2a9cbb3cf12.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
