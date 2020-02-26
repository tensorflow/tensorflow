"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "281d46b47f7191f03a8a4071c4c8d2af9409bb9d59573dc2e42f04c4fd61f1fd",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.10.3.1.3.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
