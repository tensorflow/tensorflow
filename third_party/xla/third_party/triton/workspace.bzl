"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl596935355"
    TRITON_SHA256 = "b270532407c394f86319db49dc9fc0691f56b73cb88593de2a4d79e0e8b2d38c"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b304456327.patch",
            "//third_party/triton:cl597222925.patch",
        ],
    )
