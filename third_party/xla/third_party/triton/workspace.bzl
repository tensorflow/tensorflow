"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl595147751"
    TRITON_SHA256 = "828669e624820137988796dec2075084e94b03b6bc17eaf921d337d207d96054"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b304456327.patch",
            "//third_party/triton:cl568176943.patch",
            "//third_party/triton:cl595961303.patch",
            "//third_party/triton:cl596538580.patch",
            "//third_party/triton:cl596550429.patch",
        ],
    )
