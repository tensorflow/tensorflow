"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "1627e0c27869b4098e5fa720717645c1baaf5972"
    TRITON_SHA256 = "574436dab7c65f185834bd80c1d92167bacb7471b0c25906db60686835c46e21"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl526173620.patch",
            "//third_party/triton:cl528701873.patch",
        ],
    )
