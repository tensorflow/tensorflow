# StableHLO Device Profile reference implementation

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = [":__subpackages__"],
)

cc_library(
    name = "shlo",
    srcs = [
        "src/broadcast_in_dim.cc",
        "src/clamp.cc",
        "src/compare.cc",
        "src/concatenate.cc",
        "src/dispatch.h",
        "src/elementwise_binary.cc",
        "src/elementwise_unary.cc",
        "src/iota.cc",
        "src/is_finite.cc",
        "src/select.cc",
        "src/shlo.cc",
        "src/uniform_dequantize_quantize.cc",
    ],
    hdrs = [
        "include/shlo.h",
        "src/storage.h",
        "src/util.h",
    ],
    tags = ["no_oss"],
    deps = [
        ":float",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "debug",
    srcs = [
        "src/debug.cc",
    ],
    hdrs = [
        "src/debug.h",
    ],
    tags = ["no_oss"],
    deps = [
        ":float",
        ":shlo",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "float",
    srcs = [
    ],
    hdrs = [
        "src/bf16.h",
        "src/f16.h",
        "src/has_keyword.h",
    ],
    tags = ["no_oss"],
    deps = [
    ],
)
