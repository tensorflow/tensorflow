"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl578837341"
    TRITON_SHA256 = "0d8112bb31d48b5beadbfc2e13c52770a95d3759b312b15cf26dd72e71410568"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl568176943.patch",
            "//third_party/triton:b304456327.patch",
            "//third_party/triton:cl580550344.patch",
            "//third_party/triton:cl580481414.patch",
            "//third_party/triton:cl580852372.patch",
        ],
    )
