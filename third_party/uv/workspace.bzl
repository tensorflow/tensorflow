"""Provides the repository macro to import libuv."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports libuv."""

    UV_VERSION = "v1.38.0"
    UV_SHA256 = "71344f62c5020ed3643ad0bcba98ae4d7d6037285923c5416844d7c141a3ff93"

    tf_http_archive(
        name = "uv",
        sha256 = UV_SHA256,
        strip_prefix = "libuv-{version}".format(version = UV_VERSION),
        urls = tf_mirror_urls("https://dist.libuv.org/dist/{version}/libuv-{version}.tar.gz".format(version = UV_VERSION)),
        build_file = "//third_party/uv:uv.BUILD",
    )
