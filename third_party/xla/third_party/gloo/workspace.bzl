"""Provides the repository macro to import Gloo."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Gloo."""

    GLOO_COMMIT = "5354032ea08eadd7fc4456477f7f7c6308818509"
    GLOO_SHA256 = "5759a06e6c8863c58e8ceadeb56f7c701fec89b2559ba33a103a447207bf69c7"

    tf_http_archive(
        name = "gloo",
        sha256 = GLOO_SHA256,
        strip_prefix = "gloo-{commit}".format(commit = GLOO_COMMIT),
        urls = tf_mirror_urls("https://github.com/facebookincubator/gloo/archive/{commit}.tar.gz".format(commit = GLOO_COMMIT)),
        build_file = "//third_party/gloo:gloo.BUILD",
    )
