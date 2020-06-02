"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "a0efbc5190f59b39a3ac3fab8bb95e405fd0090836f84412f230a5e168ebf4fb",
        strip_prefix = "ruy-1a8b7eabd5039cd1423b3e22e6d7241d261576dc",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/1a8b7eabd5039cd1423b3e22e6d7241d261576dc.zip",
            "https://github.com/google/ruy/archive/1a8b7eabd5039cd1423b3e22e6d7241d261576dc.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
