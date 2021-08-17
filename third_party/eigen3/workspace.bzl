"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    EIGEN_COMMIT = "5ad8b9bfe2bf75620bc89467c5cc051fc2a597df"
    EIGEN_SHA256 = "b47c1f5e54961a4289b5c690a54d78b20ae0a769a8914d81cfea764018de9b48"
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
