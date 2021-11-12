"""Loads the sobol_data library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "sobol_data",
        urls = tf_mirror_urls("https://github.com/joe-kuo/sobol_data/archive/835a7d7b1ee3bc83e575e302a985c66ec4b65249.tar.gz"),
        sha256 = "583d7b975e506c076fc579d9139530596906b9195b203d42361417e9aad79b73",
        strip_prefix = "sobol_data-835a7d7b1ee3bc83e575e302a985c66ec4b65249",
        build_file = "//third_party/sobol_data:sobol_data.BUILD",
    )
