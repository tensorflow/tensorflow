"""External versions of build rules that differ outside of Google."""

def flex_portable_tensorflow_deps():
    """Returns dependencies for building portable tensorflow in Flex delegate."""

    return [
        "//third_party/fft2d:fft2d_headers",
        "//third_party/absl/log",
        "//third_party/absl/log:check",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/absl/types:optional",
        "//third_party/eigen3",
        "//third_party/gemmlowp",
        "//third_party/icu:common",
        "//third_party/icu/data:conversion_data",
    ]

def tflite_copts_extra():
    """Defines extra compile time flags for tflite_copts(). Currently empty."""
    return []
