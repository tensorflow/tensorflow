"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-77aafa4d3b0f80feffce9ad4c718dd26751ee0e4",
        sha256 = "c32e9389f4cb079a3d1fdf0077c257650d00f824df4c079c9468b29fcafcfec7",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/77aafa4d3b0f80feffce9ad4c718dd26751ee0e4.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
