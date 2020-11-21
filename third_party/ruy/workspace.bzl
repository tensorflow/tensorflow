"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "d8f9dc52c0a52c8470e2e0b60bc16cba91853d812846c075f7ed8404990b003d",
        strip_prefix = "ruy-5bb02fbf90824c2eb6cd7418f766c593106a332b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/5bb02fbf90824c2eb6cd7418f766c593106a332b.zip",
            "https://github.com/google/ruy/archive/5bb02fbf90824c2eb6cd7418f766c593106a332b.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
