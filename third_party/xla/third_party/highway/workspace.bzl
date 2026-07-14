"""Provides the repository macro to import gutil."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports highway."""

    HIGHWAY_VERSION = "1.3.0"
    HIGHWAY_SHA256 = "07b3c1ba2c1096878a85a31a5b9b3757427af963b1141ca904db2f9f4afe0bc2"

    tf_http_archive(
        name = "com_google_highway",
        strip_prefix = "highway-{version}".format(version = HIGHWAY_VERSION),
        sha256 = HIGHWAY_SHA256,
        urls = tf_mirror_urls("https://github.com/google/highway/archive/refs/tags/{version}.tar.gz".format(version = HIGHWAY_VERSION)),
    )
