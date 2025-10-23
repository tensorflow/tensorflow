"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
<<<<<<< HEAD
    TFRT_COMMIT = "db460e7893e4accc0971876aff9b29b3382ade80"
    TFRT_SHA256 = "7065c8f508f61e8e2a579829694febe564d93dccf09f22dcc012762d5e20bc88"
=======
    TFRT_COMMIT = "4ecc3a44a32c832b748328bed3f9a599f795ca8d"
    TFRT_SHA256 = "5e81d70f9534340f7ef8e63ec43bdd5971135e48183079be50ecb3f74b1fed66"
>>>>>>> upstream/master

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = tf_mirror_urls("https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT)),
        repo_mapping = {
            "@tsl": "@local_tsl",
            "@xla": "@local_xla",
        },
        # A patch file can be provided for atomic commits to both TF and TFRT.
        # The job that bumps the TFRT_COMMIT also resets patch_file to 'None'.
        patch_file = None,
    )
