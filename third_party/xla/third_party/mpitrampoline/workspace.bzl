"""Provides the repository macro to import mpitrampoline."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports mpitrampoline."""

    MPITRAMPOLINE_COMMIT = "25efb0f7a4cd00ed82bafb8b1a6285fc50d297ed"
    MPITRAMPOLINE_SHA256 = "5a36656205c472bdb639bffebb0f014523b32dda0c2cbedd9ce7abfc9e879e84"

    tf_http_archive(
        name = "mpitrampoline",
        sha256 = MPITRAMPOLINE_SHA256,
        strip_prefix = "MPItrampoline-{commit}".format(commit = MPITRAMPOLINE_COMMIT),
        urls = tf_mirror_urls("https://github.com/eschnett/mpitrampoline/archive/{commit}.tar.gz".format(commit = MPITRAMPOLINE_COMMIT)),
        patch_file = ["//third_party/mpitrampoline:gen.patch"],
        build_file = "//third_party/mpitrampoline:mpitrampoline.BUILD",
    )
