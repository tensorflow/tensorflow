"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "ruy",
        sha256 = "243a3d0d4283c1f8e774814a4096961288a00a2662e84b3cd564afbf500bb0ad",
        strip_prefix = "ruy-c347b02c23cfc459678db6d7c230d76fac00f76d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/c347b02c23cfc459678db6d7c230d76fac00f76d.zip",
            "https://github.com/google/ruy/archive/c347b02c23cfc459678db6d7c230d76fac00f76d.zip",
        ],
        build_file = "//third_party/ruy:BUILD",
    )
