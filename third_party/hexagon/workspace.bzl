"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "tf_http_archive")

# Note: Use libhexagon_nn_skel version 1.20 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    tf_http_archive(
        name = "hexagon_nn",
        sha256 = "ade87358f2231f4d079449cbbc047c17db5a4d5cedbc230b2795adc1116e2fa5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.2.tgz",
            # Repeated to bypass 'at least two urls' check. TODO(karimnosseir): add original source of this package.
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.2.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
