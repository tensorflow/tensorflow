"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Note: Use libhexagon_nn_skel version 1.17 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "a0c011f7795e1a09eb7355be295d6442718b8565cc0e3c58a91671dde2bc99fb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.17.0.0.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
