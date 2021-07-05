"""Provides the repository macro to import TFRT."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports TFRT."""

    # Attention: tools parse and update these lines.
    TFRT_COMMIT = "fb07b1209b423953564dff7536deef2cd4f0902a"
    TFRT_SHA256 = "f0d211c9882afd6db15be12f6e6c5696f23f1cd35e64ae2d9959ef7473a04b96"

    tf_http_archive(
        name = "tf_runtime",
        sha256 = TFRT_SHA256,
        strip_prefix = "runtime-{commit}".format(commit = TFRT_COMMIT),
        urls = [
            "http://mirror.tensorflow.org/github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
            "https://github.com/tensorflow/runtime/archive/{commit}.tar.gz".format(commit = TFRT_COMMIT),
        ],
    )
