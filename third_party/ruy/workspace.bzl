"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "89b8b56b4e1db894e75a0abed8f69757b37c23dde6e64bfb186656197771138a",
        strip_prefix = "ruy-388ffd28ba00ffb9aacbe538225165c02ea33ee3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/388ffd28ba00ffb9aacbe538225165c02ea33ee3.zip",
            "https://github.com/google/ruy/archive/388ffd28ba00ffb9aacbe538225165c02ea33ee3.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
