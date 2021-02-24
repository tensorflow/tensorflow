"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "90ee821c563fa20db4d64d6991ddca256d5c52f2"
    EIGEN_SHA256 = "d76992f1972e4ff270221c7ee8125610a8e02bb46708a7295ee646e99287083b"

    tf_http_archive(
        name = name,
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )
