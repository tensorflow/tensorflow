"""Provides the repo macro to import pybind11_bazel.

pybind11_bazel requires pybind11 (which is loaded in another rule).
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-2.13.6",
        sha256 = "cae680670bfa6e82703c03f2a3c995408cdcbf43616d7bdd198ef45d3c327731",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_bazel/archive/v2.13.6.tar.gz"),
    )
