"""Provides the repository macro to import Gloo."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Gloo."""

    GLOO_COMMIT = "54cbae0d3a67fa890b4c3d9ee162b7860315e341"
    GLOO_SHA256 = "61089361dbdbc9d6f75e297148369b13f615a3e6b78de1be56cce74ca2f64940"

    tf_http_archive(
        name = "gloo",
        sha256 = GLOO_SHA256,
        strip_prefix = "gloo-{commit}".format(commit = GLOO_COMMIT),
        urls = tf_mirror_urls("https://github.com/facebookincubator/gloo/archive/{commit}.tar.gz".format(commit = GLOO_COMMIT)),
        build_file = "//third_party/gloo:gloo.BUILD",
    )
