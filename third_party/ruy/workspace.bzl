"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "d5e703913c9e8f0196d83cc4113ecaae4bcae52181f05836890f16aad402fea4",
        strip_prefix = "ruy-51b518e755dd3da37a79d16972b76d3baedac22d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/51b518e755dd3da37a79d16972b76d3baedac22d.zip",
            "https://github.com/google/ruy/archive/51b518e755dd3da37a79d16972b76d3baedac22d.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
