"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-790d7a083520398268d92d0bd61cf85dfa32ee98",
        sha256 = "147cc89904375dcd0b0d664a2b72dfadbb02058800ad8cba84516094bc406208",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/790d7a083520398268d92d0bd61cf85dfa32ee98.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
