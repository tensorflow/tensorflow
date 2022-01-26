"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content.
    EIGEN_COMMIT = "7db0ac977acf276fb0817cfb89e490cdbae0ab56"
    EIGEN_SHA256 = "f4697dcc5f0545cf9ab70be8f6b313bdc31c0405f0ed30f8967bc02e7d7aa12a"

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "4553df1eb1287c53d11a661bec282d37fd028864a3a324cf9b085c9d99b6cf55",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
