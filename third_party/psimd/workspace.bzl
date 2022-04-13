"""Loads the psimd library, used by XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "psimd",
        strip_prefix = "psimd-072586a71b55b7f8c584153d223e95687148a900",
        sha256 = "dc615342bcbe51ca885323e51b68b90ed9bb9fa7df0f4419dbfa0297d5e837b7",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/psimd/archive/072586a71b55b7f8c584153d223e95687148a900.zip"),
        build_file = "//third_party/psimd:psimd.BUILD",
    )
