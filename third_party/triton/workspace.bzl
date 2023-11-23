"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl584018112"
    TRITON_SHA256 = "a0f2461af9fbcf576cef08e0b83ab7a1caa3cfe2041c60b2809cbd495ff14f08"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b304456327.patch",
            "//third_party/triton:cl568176943.patch",
            "//third_party/triton:cl584230333.patch",
        ],
    )
