"""External versions of build rules that differ outside of Google."""

def flex_portable_tensorflow_deps():
    """Returns dependencies for building portable tensorflow in Flex delegate."""

    return [
        "//third_party/fft2d:fft2d_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
        "@gemmlowp",
        "@icu//:common",
        "//third_party/icu/data:conversion_data",
    ]

def tflite_copts_extra():
    """Defines extra compile time flags for tflite_copts(). Currently empty."""
    return []
