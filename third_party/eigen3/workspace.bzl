"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    EIGEN_COMMIT = "0ae94456a0e6dd5e20ca65ba2f405964f6931faf"
    EIGEN_SHA256 = "f62c55c96e13a1ac635c7c7a78e21bfb2221ab78f0ade891e4785a44503caf99"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/eigen.cmake)

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
