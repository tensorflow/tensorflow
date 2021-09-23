"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    EIGEN_COMMIT = "7792b1e909a98703181aecb8810b4b654004c25d"
    EIGEN_SHA256 = "517a02cd24362fd9284ce61e8baeeabb4fee181941752da47eb4c07a52d38d4d"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/eigen.cmake)

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
