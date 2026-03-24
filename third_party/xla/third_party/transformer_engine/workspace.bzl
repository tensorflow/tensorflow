"""Loads the TransformerEngine library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "transformer_engine",
        strip_prefix = "TransformerEngine-2.5",
        sha256 = "ee52ee9e43e44edc8598bc3d111eedc2445c9ebfe78a1fcab6f5c4c887020b72",
        urls = tf_mirror_urls("https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v2.5.tar.gz"),
        build_file = "//third_party/transformer_engine:transformer_engine.BUILD",
        patch_file = ["//third_party/transformer_engine:transformer_engine.patch"],
    )
