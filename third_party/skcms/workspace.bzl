"""Point to the SkCMS repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "skcms",
        strip_prefix = "skcms-df39e60213b19b6098d8d3530c8c385c3b99a5e8",
        sha256 = "2a50a631c513e4b7d591873b8893c5c96b1e4285c5c06d4e2239c4a8d4b3b84f",
        urls = tf_mirror_urls("https://github.com/google/skcms/archive/df39e60213b19b6098d8d3530c8c385c3b99a5e8.zip"),
        build_file = "//third_party/skcms:skcms.BUILD.bazel",
    )
