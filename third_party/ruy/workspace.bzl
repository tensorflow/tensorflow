"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "9177fbb25e3875a82e171e9e9b70d65d8d31ffa41eea3e479b6a27c8767d1f5d",
        strip_prefix = "ruy-b68dcd87137abe5cef13cc4d15bfa541874cbd96",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/b68dcd87137abe5cef13cc4d15bfa541874cbd96.zip",
            "https://github.com/google/ruy/archive/b68dcd87137abe5cef13cc4d15bfa541874cbd96.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
