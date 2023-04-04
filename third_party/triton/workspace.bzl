"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "9f82617fb910149068ed45d4144ce204d16accf1"
    TRITON_SHA256 = "42b7951ce23d9487401d6f89f33f7f471596b0cd98626a65f986a80f9b9c6f5f"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        patch_file = [],
    )
