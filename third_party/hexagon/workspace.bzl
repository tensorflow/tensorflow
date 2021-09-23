"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

# Note: Use libhexagon_nn_skel version 1.20 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    tf_http_archive(
        name = "hexagon_nn",
        sha256 = "6eaf6d8eabfcb3486753c68f22fd6c1eabf8fae28bf52e4ebea815e9daf67257",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.3.tgz",
            # Repeated to bypass 'at least two urls' check. TODO(karimnosseir): add original source of this package.
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.3.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
