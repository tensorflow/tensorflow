"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl547802044"
    TRITON_SHA256 = "08c50c1d2500ac9bcc6f1ac0ce185a277d15e486e916da03c4a1613873d52299"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl536931041.patch",
            "//third_party/triton:cl547802044.patch",
            "//third_party/triton:cl550499635.patch",
            "//third_party/triton:cl551490193.patch",
            "//third_party/triton:cl553072809.patch",
            "//third_party/triton:cl553418757.patch",
        ],
    )
