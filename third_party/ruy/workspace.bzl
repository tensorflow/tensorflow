"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "fe8345f521bb378745ebdd0f8c5937414849936851d2ec2609774eb2d7098e54",
        strip_prefix = "ruy-9f53ba413e6fc879236dcaa3e008915973d67a4f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
            "https://github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
