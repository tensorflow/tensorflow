"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "ruy",
        sha256 = "525de68739faa23eeea674596607a3eea7ca4425be2962b26775158e084c1036",
        strip_prefix = "ruy-d37128311b445e758136b8602d1bbd2a755e115d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip",
            "https://github.com/google/ruy/archive/d37128311b445e758136b8602d1bbd2a755e115d.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
