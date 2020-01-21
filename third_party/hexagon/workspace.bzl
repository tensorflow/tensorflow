"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "e972f86eb8bcfb1ee93ff3dc7aa4518948e3941b5ea0945f5c9307b2d3334225",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.10.3.1.0.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
