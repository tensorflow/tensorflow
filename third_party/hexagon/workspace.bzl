"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "4cbf3c18834e24b1f64cc507f9c2f22b4fe576c6ff938d55faced5d8f1bddf62",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.10.3.1.2.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
