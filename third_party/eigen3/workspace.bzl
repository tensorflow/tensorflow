"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo(name):
    """Imports Eigen."""
    third_party_http_archive(
        name = name,
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "d76992f1972e4ff270221c7ee8125610a8e02bb46708a7295ee646e99287083b",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-90ee821c563fa20db4d64d6991ddca256d5c52f2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/90ee821c563fa20db4d64d6991ddca256d5c52f2/eigen-90ee821c563fa20db4d64d6991ddca256d5c52f2.tar.gz",
            "https://gitlab.com/libeigen/eigen/-/archive/90ee821c563fa20db4d64d6991ddca256d5c52f2/eigen-90ee821c563fa20db4d64d6991ddca256d5c52f2.tar.gz",
        ],
    )
