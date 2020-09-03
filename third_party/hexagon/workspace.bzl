"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Note: Use libhexagon_nn_skel version 1.20 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "2b0e29a061f389ad52054c12fcae38991b5f731d7a05770c7ac421433ed17cc2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.0.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
