"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

# Note: Use libhexagon_nn_skel version 1.20.0.1 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    tf_http_archive(
        name = "hexagon_nn",
        sha256 = "f577b4c150b72e11e9dfb3f9d14f9772ba8fe460f7d65c84a7327ea9bef44d8e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.9.tgz",
            # Repeated to bypass 'at least two urls' check. TODO(karimnosseir): add original source of this package.
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.9.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
