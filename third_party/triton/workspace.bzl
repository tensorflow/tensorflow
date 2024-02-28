"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl608559313"
    TRITON_SHA256 = "d37c0a2921f756cb355dc7ea7e91ea708cef867117edff37106f5a947c5a5a38"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl607293980.patch",  # long standing :(
            "//third_party/triton:cl610393680.patch",
            "//third_party/triton:cl610484237.patch",
            "//third_party/triton:cl610740193.patch",
        ],
    )
