"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "43aff3de4f0924852b634dc5e72f0ae3b0e3957b9d514ca4c5ae03b09b5a3884",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.10.3.1.1.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
