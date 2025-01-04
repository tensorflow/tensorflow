"""Provides the repo macro to import pybind11_bazel.

pybind11_bazel requires pybind11 (which is loaded in another rule).
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    PB_COMMIT = "ea71d9764f5e62708a4c3b40d3994e7e8d422324"
    PB_SHA256 = "5ab4da506efbeb493a408420bc9b095409f86d6d754351aee2a3c26a71954882"
    tf_http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PB_COMMIT),
        sha256 = PB_SHA256,
        patch_file = ["//third_party/pybind11_bazel:pybind11_bazel.patch"],
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_bazel/archive/{commit}.tar.gz".format(commit = PB_COMMIT)),
    )
