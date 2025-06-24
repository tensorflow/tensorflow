"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-1.1",
        sha256 = "2e3b94b55825c240cc58e6721e15b449978cbae21a2a4caa23058b0157ee2fb3",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/refs/tags/v1.1.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
