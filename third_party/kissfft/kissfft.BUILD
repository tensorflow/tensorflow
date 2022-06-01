package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["COPYING"])

cc_library(
    name = "kiss_fftr_16",
    srcs = [
        "kfc.c",
        "kiss_fft.c",
        "kiss_fftnd.c",
        "kiss_fftndr.c",
        "kiss_fftr.c",
    ],
    hdrs = [
        "_kiss_fft_guts.h",
        "kfc.h",
        "kiss_fft.h",
        "kiss_fft_log.h",
        "kiss_fftnd.h",
        "kiss_fftndr.h",
        "kiss_fftr.h",
    ],
    copts = [
        "-DFIXED_POINT=16",
    ],
)
