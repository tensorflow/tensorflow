"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl596935355"
    TRITON_SHA256 = "66bac24a0443655c8f4efe542bc78c0c34cf8756baa746e61604cf26bd9fae5e"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:b304456327.patch",
            "//third_party/triton:cl597183646.patch",
            "//third_party/triton:cl597222925.patch",
            "//third_party/triton:cl598847928.patch",
            "//third_party/triton:cl599125711.patch",
        ],
    )
