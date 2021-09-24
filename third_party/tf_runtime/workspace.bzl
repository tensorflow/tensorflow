"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "d82a7d056839a2d2c934f912e7ab7f45dca23bb7"
    TFRT_SHA256 = "4d1b7a438a7aeb9da3edab8f2e2c5aaae9047f25f8b2a361c555562fa7494dad"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = [
            "http://mirror.tensorflow.org/github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
            "https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
        ],
        # A patch file can be provided for atomic commits to both TF and TFRT.
        # The job that bumps the TFRT_COMMIT also resets patch_file to 'None'.
        patch_file = None,
    )
