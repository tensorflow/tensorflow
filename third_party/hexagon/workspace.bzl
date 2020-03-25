"""Loads the Hexagon NN Header files library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Note: Use libhexagon_nn_skel version 1.14 Only with the current version.
# This comment will be updated with compatible version.
def repo():
    third_party_http_archive(
        name = "hexagon_nn",
        sha256 = "5a0e72b20a47d826c3f0437a2fbc099bb214413244ab42979c9832fefe15ff63",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.14.0.0.tgz",
        ],
        build_file = "//third_party/hexagon:BUILD",
    )
