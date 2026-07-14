"""Loads the coremltools library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "coremltools",
        sha256 = "37d4d141718c70102f763363a8b018191882a179f4ce5291168d066a84d01c9d",
        strip_prefix = "coremltools-8.0",
        build_file = "//third_party/coremltools:coremltools.BUILD",
        urls = tf_mirror_urls("https://github.com/apple/coremltools/archive/8.0.tar.gz"),
    )
