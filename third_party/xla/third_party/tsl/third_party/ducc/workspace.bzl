"""Distinctly Useful Code Collection (DUCC) - CPU FFT Module"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    DUCC_COMMIT = "aa46a4c21e440b3d416c16eca3c96df19c74f316"
    DUCC_SHA256 = "077cf4bd0bd7eddaa6649a024285fff96e2662c5e6f2fb6ed5c5771f9de093f3"
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
