"""Provides the repository macro to import mpitrampoline."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports mpitrampoline."""

    MPITRAMPOLINE_COMMIT = "de3d086af6b18f6aef5ff397be054d7f66bae052"
    MPITRAMPOLINE_SHA256 = "e5dce8be575b99d20c65831549629449e9b5045f389f58c624c280d614fa0fd9"

    tf_http_archive(
        name = "mpitrampoline",
        sha256 = MPITRAMPOLINE_SHA256,
        strip_prefix = "MPItrampoline-{commit}".format(commit = MPITRAMPOLINE_COMMIT),
        urls = tf_mirror_urls("https://github.com/eschnett/mpitrampoline/archive/{commit}.tar.gz".format(commit = MPITRAMPOLINE_COMMIT)),
        patch_file = ["//third_party/mpitrampoline:gen.patch"],
        build_file = "//third_party/mpitrampoline:mpitrampoline.BUILD",
    )
