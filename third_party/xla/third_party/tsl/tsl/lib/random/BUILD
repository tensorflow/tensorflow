load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

default_visibility = [
    "//tsl/lib/io:__pkg__",
    # tensorflow/core/platform/random aliases this package
    "//tensorflow/core/lib/random:__pkg__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(default_visibility),
    licenses = ["notice"],
)

cc_library(
    name = "exact_uniform_int",
    hdrs = ["exact_uniform_int.h"],
)

cc_library(
    name = "philox",
    srcs = [
        "distribution_sampler.cc",
        "random_distributions.cc",
        "simple_philox.cc",
    ],
    hdrs = [
        "distribution_sampler.h",
        "random_distributions.h",
        "simple_philox.h",
    ],
    visibility = internal_visibility(default_visibility),
    deps = [
        ":exact_uniform_int",
        ":philox_random",
        ":random_distributions_utils",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "random_distributions_utils",
    hdrs = ["random_distributions_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core/lib/random:__pkg__",
        "//tensorflow/lite:__subpackages__",
    ]),
    deps = [":philox_random"],
)

cc_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core/lib/random:__pkg__",
        "//tensorflow/lite:__subpackages__",
    ]),
)

cc_library(
    name = "philox_random_test_utils",
    testonly = True,
    hdrs = ["philox_random_test_utils.h"],
    deps = [
        ":philox_random",
        "//tsl/platform:logging",
        "//tsl/platform:random",
    ],
)

cc_library(
    name = "weighted_picker",
    srcs = ["weighted_picker.cc"],
    hdrs = ["weighted_picker.h"],
    deps = [
        ":philox",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
    ],
    alwayslink = 1,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "distribution_sampler.cc",
        "distribution_sampler.h",
        "exact_uniform_int.h",
        "philox_random.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "simple_philox.cc",
        "simple_philox.h",
        "weighted_picker.cc",
        "weighted_picker.h",
    ],
)

filegroup(
    name = "legacy_lib_random_headers",
    srcs = [
        "distribution_sampler.h",
        "philox_random.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "simple_philox.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/random:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_random_headers",
    srcs = [
        "random_distributions.h",
        "random_distributions_utils.h",
        "weighted_picker.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/random:__pkg__"]),
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "philox_random_test_utils.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/random:__pkg__"]),
)

filegroup(
    name = "legacy_lib_random_all_headers",
    srcs = [
        "distribution_sampler.h",
        "exact_uniform_int.h",
        "philox_random.h",
        "philox_random_test_utils.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "simple_philox.h",
        "weighted_picker.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/random:__pkg__"]),
)

tsl_cc_test(
    name = "distribution_sampler_test",
    size = "small",
    srcs = ["distribution_sampler_test.cc"],
    deps = [
        ":philox",
        "//tsl/platform:macros",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "philox_random_test",
    size = "small",
    srcs = ["philox_random_test.cc"],
    deps = [
        ":philox",
        ":philox_random",
        ":philox_random_test_utils",
        "//tsl/platform:logging",
        "//tsl/platform:random",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "random_distributions_test",
    srcs = ["random_distributions_test.cc"],
    tags = ["optonly"],
    deps = [
        ":philox",
        ":philox_random",
        ":philox_random_test_utils",
        "//tsl/lib/math:math_util",
        "//tsl/platform:logging",
        "//tsl/platform:random",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "simple_philox_test",
    size = "small",
    srcs = ["simple_philox_test.cc"],
    deps = [
        ":philox",
        "//tsl/platform:logging",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "weighted_picker_test",
    size = "medium",
    srcs = ["weighted_picker_test.cc"],
    deps = [
        ":philox",
        ":weighted_picker",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
    ],
)
