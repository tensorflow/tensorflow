"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "7f09d3487de97882585a72819ec7e0f9100c7121"
    EIGEN_SHA256 = "6ae281a5a32d0f4185856e790c06f58858ffc16594483281621746ffb74d88a2"

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )
