"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-1.9.2",
        sha256 = "149a3da40b0a988513d8cf5e71db3037373823505a3c92f87b988c92d7e0ab34",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v1.9.2.tar.gz"),
        build_file = "//third_party/nanobind:nanobind.BUILD",
        patch_file = [
            "//third_party/nanobind:pr438.patch",  # Remove when updating to nanobind 2.0.0.
            "//third_party/nanobind:pr461.patch",  # Remove when updating to nanobind 2.0.0.
        ],
    )
