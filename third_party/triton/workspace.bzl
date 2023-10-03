"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "cl555471166"
    TRITON_SHA256 = "da847562e615c0c5c2316a514f9aa6dd74b36fe781a4eb3f0d23105b2c514e76"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = [
            "//third_party/triton:cl566656131.patch",
            "//third_party/triton:cl536931041.patch",
            "//third_party/triton:cl555471166.patch",
            "//third_party/triton:cl561859552.patch",
            "//third_party/triton:cl564681263.patch",
            "//third_party/triton:msvc_fixes.patch",
            "//third_party/triton:cl547477882.patch",
            "//third_party/triton:cl565616678.patch",
            "//third_party/triton:cl565664892.patch",
            "//third_party/triton:cl566223642.patch",
            "//third_party/triton:cl568240805.patch",
            "//third_party/triton:cl568793052.patch",
        ],
    )
