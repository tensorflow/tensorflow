"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl580208989"
    TRITON_SHA256 = "bcf6e99a73c8797720325b0f2e48447cdae7f68c53c68bfe04c39104db542562"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b311157761.patch",
            "//third_party/triton:cl568176943.patch",
            "//third_party/triton:b304456327.patch",
        ],
    )
