"""Point to the Brotli repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_brotli",
        strip_prefix = "brotli-1.2.0",
        sha256 = "816c96e8e8f193b40151dad7e8ff37b1221d019dbcb9c35cd3fadbfe6477dfec",
        urls = tf_mirror_urls("https://github.com/google/brotli/archive/refs/tags/v1.2.0.tar.gz"),
    )
