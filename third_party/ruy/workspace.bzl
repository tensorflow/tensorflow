"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "ac6d71df496a20043252f451d82a01636bb8bba9c3d6b5dc9fadadaffa392751",
        strip_prefix = "ruy-91d62808498cea7ccb48aa59181e218b4ad05701",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/91d62808498cea7ccb48aa59181e218b4ad05701.zip",
            "https://github.com/google/ruy/archive/91d62808498cea7ccb48aa59181e218b4ad05701.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
