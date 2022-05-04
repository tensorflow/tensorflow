"""Provides the repo macro to import pybind11_abseil.

pybind11_abseil requires pybind11 (which is loaded in another rule) and pybind11_bazel.
See https://github.com/pybind/pybind11_abseil#installation.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports pybind11_abseil."""
    PA_COMMIT = "d9614e4ea46b411d02674305245cba75cd91c1c6"
    PA_SHA256 = "a2b5509dc1c344954fc2f1ba1d277afae84167691c0daad59b6da71886d1f276"
    tf_http_archive(
        name = "pybind11_abseil",
        sha256 = PA_SHA256,
        strip_prefix = "pybind11_abseil-{commit}".format(commit = PA_COMMIT),
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_abseil/archive/{commit}.tar.gz".format(commit = PA_COMMIT)),
        build_file = "//third_party/pybind11_abseil:BUILD",
    )

    # pybind11_bazel is a dependency of pybind11_abseil.
    PB_COMMIT = "26973c0ff320cb4b39e45bc3e4297b82bc3a6c09"
    PB_SHA256 = "8f546c03bdd55d0e88cb491ddfbabe5aeb087f87de2fbf441391d70483affe39"
    tf_http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PB_COMMIT),
        sha256 = PB_SHA256,
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.tar.gz"),
    )
