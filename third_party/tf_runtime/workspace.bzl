"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "2ed451b72dc9e57fd0a4f9526df852a9a02d8f2d"
    TFRT_SHA256 = "0fc6ff00ad2ac6feaaeff20e6b9d5b73c4c5063a611e5e2d284f150435167005"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = [
            "http://mirror.tensorflow.org/github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
            "https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
        ],
    )
