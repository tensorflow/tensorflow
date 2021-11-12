"""Loads a lightweight subset of the ICU library for Unicode processing."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "icu",
        strip_prefix = "icu-release-64-2",
        sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
        urls = tf_mirror_urls("https://github.com/unicode-org/icu/archive/release-64-2.zip"),
        build_file = "//third_party/icu:icu.BUILD",
        system_build_file = "//third_party/icu:BUILD.system",
        patch_file = "//third_party/icu:udata.patch",
    )
