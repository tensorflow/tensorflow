"""Point to the libwebp repo on GitHub."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Use the same libwebp release as tensorstore
    tf_http_archive(
        name = "libwebp",
        strip_prefix = "libwebp-1.4.0",
        sha256 = "12af50c45530f0a292d39a88d952637e43fb2d4ab1883c44ae729840f7273381",
        urls = tf_mirror_urls("https://github.com/webmproject/libwebp/archive/v1.4.0.tar.gz"),
        build_file = "//third_party/libwebp:libwebp.BUILD.bazel",
    )
