"""Loads a lightweight subset of the ICU library for Unicode processing."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# NOTE: If you upgrade this, generate the data files by following the
# instructions in third_party/icu/data/BUILD
def repo():
    tf_http_archive(
        name = "icu",
        strip_prefix = "icu-release-69-1",
        sha256 = "3144e17a612dda145aa0e4acb3caa27a5dae4e26edced64bc351c43d5004af53",
        urls = tf_mirror_urls("https://github.com/unicode-org/icu/archive/release-69-1.zip"),
        build_file = "//third_party/icu:icu.BUILD",
        system_build_file = "//third_party/icu:BUILD.system",
        patch_file = ["//third_party/icu:udata.patch"],
    )
