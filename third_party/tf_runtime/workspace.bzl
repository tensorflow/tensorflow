"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "23364be0516cf94568b5b4968a8933a984cf2c45"
    TFRT_SHA256 = "a9e14d470e1db3fb5ae3c410218ef4131495ad1fd46cc4e736a41e7f06807974"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = [
            "http://mirror.tensorflow.org/github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
            "https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
        ],
    )
