"""Provides the repo macro to import pybind11_abseil.

pybind11_abseil requires pybind11 (which is loaded in another rule) and pybind11_bazel.
See https://github.com/pybind/pybind11_abseil#installation.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports pybind11_abseil."""

    # Updating past this commit causes failures with custom ops:
    #   ModuleNotFoundError: No module named 'pybind11_abseil'
    PA_COMMIT = "13d4f99d5309df3d5afa80fe2ae332d7a2a64c6b"
    PA_SHA256 = "c6d0c6784e4d5681919731f1fa86e0b7cd010e770115bdb3a0285b3939ef2394"
    tf_http_archive(
        name = "pybind11_abseil",
        sha256 = PA_SHA256,
        strip_prefix = "pybind11_abseil-{commit}".format(commit = PA_COMMIT),
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_abseil/archive/{commit}.tar.gz".format(commit = PA_COMMIT)),
        patch_file = ["//third_party/pybind11_abseil:remove_license.patch"],
    )
