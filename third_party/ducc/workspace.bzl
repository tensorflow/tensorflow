"""Distinctly Useful Code Collection (DUCC) - CPU FFT Module"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    DUCC_COMMIT = "15246e40cfa880c079606ce49bbf629b07fcf9cb"
    DUCC_SHA256 = "82d961c0227b6842f2fcac30dd178f6fd278b54e0405d3e8c977ed6981ea0e70"
    tf_http_archive(
        name = "ducc",
        strip_prefix = "ducc-{commit}".format(commit = DUCC_COMMIT),
        sha256 = DUCC_SHA256,
        urls = tf_mirror_urls("https://gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/{commit}/ducc-{commit}.tar.gz".format(commit = DUCC_COMMIT)),
        build_file = "//third_party/ducc:ducc.BUILD",
        link_files = {
            "//third_party/ducc:ducc0_custom_lowlevel_threading.h": "google/ducc0_custom_lowlevel_threading.h",
            "//third_party/ducc:fft.h": "google/fft.h",
            "//third_party/ducc:fft.cc": "google/fft.cc",
            "//third_party/ducc:threading.cc": "google/threading.cc",
            "//third_party/ducc:threading.h": "google/threading.h",
        },
    )
