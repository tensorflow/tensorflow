"""Provides the repository macro to import kokkos."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports kokkos."""

    KOKKOS_VERSION = "4.4.01"
    KOKKOS_SHA256 = "3f7096d17eaaa4004c7497ac082bf1ae3ff47b5104149e54af021a89414c3682"

    tf_http_archive(
        name = "kokkos",
        sha256 = KOKKOS_SHA256,
        strip_prefix = "kokkos-{version}".format(version = KOKKOS_VERSION),
        urls = tf_mirror_urls("https://github.com/kokkos/kokkos/archive/refs/tags/{version}.tar.gz".format(version = KOKKOS_VERSION)),
        build_file = "//third_party/kokkos:kokkos.BUILD",
        patch_file = ["//third_party/kokkos:smoke_test.cc.patch"],
    )
