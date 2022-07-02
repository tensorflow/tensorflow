"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "b14df3cdcf98a6c21a9e7434a793be38b0152fef"
    TFRT_SHA256 = "b1cdcf58c7b8597a159536a22f5eb51c598df7e2a936b13a4984c2c235d40b26"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = tf_mirror_urls("https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT)),
        # A patch file can be provided for atomic commits to both TF and TFRT.
        # The job that bumps the TFRT_COMMIT also resets patch_file to 'None'.
        patch_file = None,
    )
