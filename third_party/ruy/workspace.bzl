"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "28331222625e677be004e96da5e9a1cc9d65187d04d70d1ab2ca58445461ecbc",
        strip_prefix = "ruy-4790797d11a81f96baf24f3731fd3ca44c2c5f8b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/4790797d11a81f96baf24f3731fd3ca44c2c5f8b.zip",
            "https://github.com/google/ruy/archive/4790797d11a81f96baf24f3731fd3ca44c2c5f8b.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
