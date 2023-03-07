"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "3c1b5862866c72d8bf8d34290b6981fd40c0a30b"
    TRITON_SHA256 = "4605009863a5201e5250353b1e73b0edeee81a6bca43df2670e6496d76b5841e"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [],
    )
