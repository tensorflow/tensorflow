"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "2c4c4edd6c8eab567931d5191d09c2494038aec87fc47453034e258ef034ca91",
        strip_prefix = "ruy-93fdb9e66f5c53e5290ee4f74d4e7b3be4e2afc5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/93fdb9e66f5c53e5290ee4f74d4e7b3be4e2afc5.zip",
            "https://github.com/google/ruy/archive/93fdb9e66f5c53e5290ee4f74d4e7b3be4e2afc5.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
