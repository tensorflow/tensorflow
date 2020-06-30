"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "8fd4adeeff4f29796bf7cdda64806ec0495a2435361569f02afe3fe33406f07c",
        strip_prefix = "ruy-34ea9f4993955fa1ff4eb58e504421806b7f2e8f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/34ea9f4993955fa1ff4eb58e504421806b7f2e8f.zip",
            "https://github.com/google/ruy/archive/34ea9f4993955fa1ff4eb58e504421806b7f2e8f.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
