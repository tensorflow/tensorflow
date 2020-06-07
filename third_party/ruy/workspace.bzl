"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "b21524de00c63b3d5683b42557f78452e791cf77fddb2e63f9bcba1f7bd99093",
        strip_prefix = "ruy-1b313682ef8b8fc8ed08719c610d1c3503b016bf",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/1b313682ef8b8fc8ed08719c610d1c3503b016bf.zip",
            "https://github.com/google/ruy/archive/1b313682ef8b8fc8ed08719c610d1c3503b016bf.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
