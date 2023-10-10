"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "70637966e2ec9afccc2cf4d51ed2391172b1b9c5"
    TFRT_SHA256 = "e3c5911cd4a70b01de980479b3c5aa45235240bf48a07cd9c1016cd140917fd3"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = tf_mirror_urls("https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT)),
        # A patch file can be provided for atomic commits to both TF and TFRT.
        # The job that bumps the TFRT_COMMIT also resets patch_file to 'None'.
        patch_file = ["//third_party/tf_runtime:tf_runtime_clangcl.patch"],
    )
