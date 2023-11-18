"""Distinctly Useful Code Collection (DUCC) - CPU FFT Module"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    DUCC_COMMIT = "3d28aadfd8bb0219e3df188613dbbcdfffccc3cd"
    DUCC_SHA256 = "eb044dd11374ed894d67081109d4aa7ed55c29fe3286b116f13db70da6af336c"
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
