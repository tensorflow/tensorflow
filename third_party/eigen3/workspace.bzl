"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    EIGEN_COMMIT = "853a5c4b843a3f1de5de2a25429eefd62dbd153a"
    EIGEN_SHA256 = "767d07ad5bece4a961dc4025147418a17f8ce1a24fa4ef685876a275a5e65e7c"

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
