"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "ee675492d4bf787e89025491b447211f6ea6cdcd"
    TRITON_SHA256 = "59f594b55cc2054034840203f4889abfd204a2d34a7d167279b18e6d5c05575c"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl523184810.patch",
            "//third_party/triton:cl523448105.patch",
            "//third_party/triton:cl526173620.patch",
        ],
    )
