"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "51c1492196cdd6fc524dd8b539de5d644bbb436699fab3908585a575e347c789",
        strip_prefix = "ruy-4bdb31ab484e624deef9620ecde2156ca17f6567",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/4bdb31ab484e624deef9620ecde2156ca17f6567.zip",
            "https://github.com/google/ruy/archive/4bdb31ab484e624deef9620ecde2156ca17f6567.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
