"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: TensorFlow Lite CMake build uses this variable, update only the hash content.
    EIGEN_COMMIT = "008ff3483a8c5604639e1c4d204eae30ad737af6"
    EIGEN_SHA256 = "e1dd31ce174c3d26fbe38388f64b09d2adbd7557a59e90e6f545a288cc1755fc"

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "4553df1eb1287c53d11a661bec282d37fd028864a3a324cf9b085c9d99b6cf55",
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
