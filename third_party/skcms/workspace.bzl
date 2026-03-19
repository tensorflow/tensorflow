"""Point to the skcms repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "skcms",
        strip_prefix = "skcms-2025_09_16",
        sha256 = "08c45dff8ede1b56a6e7d1e9fdaf113dd91ab28ddcbcf696229b683e5e9af45a",
        urls = tf_mirror_urls("https://github.com/google/skcms/archive/refs/tags/2025_09_16.tar.gz"),
        build_file = "//third_party/skcms:skcms.BUILD.bazel",
    )
