"""Provides the repo macro to import pybind11_protobuf.

pybind11_protobuf provides automatic bindings for protobuf messages in pybind11.
See https://github.com/pybind/pybind11_protobuf for more information.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports pybind11_protobuf."""

    tf_http_archive(
        name = "pybind11_protobuf",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.zip"),
        sha256 = "3cf7bf0f23954c5ce6c37f0a215f506efa3035ca06e3b390d67f4cbe684dce23",
        strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
        patch_file = [
            "//third_party/pybind11_protobuf:protobuf.patch",
            "//third_party/pybind11_protobuf:remove_license.patch",
        ],
    )