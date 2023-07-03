"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl539572190"
    TRITON_SHA256 = "382a50741544c63328b6b15a9740b680cd5e7b83d44d85e04faa462d15d6cfc2"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl536931041.patch",
            "//third_party/triton:cl540377639.patch",
            "//third_party/triton:cl543665615.patch",
        ],
    )
