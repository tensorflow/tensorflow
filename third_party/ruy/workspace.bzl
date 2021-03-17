"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "ruy",
        sha256 = "da5ec0cc07472bdb21589b0b51c8f3d7f75d2ed6230b794912adf213838d289a",
        strip_prefix = "ruy-54774a7a2cf85963777289193629d4bd42de4a59",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/54774a7a2cf85963777289193629d4bd42de4a59.zip",
            "https://github.com/google/ruy/archive/54774a7a2cf85963777289193629d4bd42de4a59.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
