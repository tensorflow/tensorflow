"""Provides the repo macro to import pybind11_bazel.

pybind11_bazel requires pybind11 (which is loaded in another rule).
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    PB_COMMIT = "72cbbf1fbc830e487e3012862b7b720001b70672"
    PB_SHA256 = "516c1b3a10d87740d2b7de6f121f8e19dde2c372ecbfe59aef44cd1872c10395"
    tf_http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-{commit}".format(commit = PB_COMMIT),
        sha256 = PB_SHA256,
        patch_file = ["//third_party/pybind11_bazel:pybind11_bazel.patch"],
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_bazel/archive/{commit}.tar.gz".format(commit = PB_COMMIT)),
    )
