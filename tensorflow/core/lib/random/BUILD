load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/platform/cloud depends on random.h
        "//tensorflow/core/platform:__subpackages__",
        # All of the following packages depend on random/simple_philox.h for tests
        "//tensorflow/core/lib/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
        "//tensorflow/core/lib/io:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "exact_uniform_int",
    hdrs = ["exact_uniform_int.h"],
    deps = ["//tensorflow/tsl/lib/random:exact_uniform_int"],
)

cc_library(
    name = "philox",
    hdrs = [
        "distribution_sampler.h",
        "random_distributions.h",
        "simple_philox.h",
    ],
    deps = [
        ":exact_uniform_int",
        ":philox_random",
        ":random_distributions_utils",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/random:philox",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "random_distributions_utils",
    hdrs = ["random_distributions_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        ":philox_random",
        "//tensorflow/tsl/lib/random:random_distributions_utils",
    ],
)

cc_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/lite:__subpackages__",
    ],
    deps = ["//tensorflow/tsl/lib/random:philox_random"],
)

cc_library(
    name = "random",
    hdrs = ["random.h"],
    visibility = [
        "//tensorflow/c/eager:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/grappler/optimizers:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
        "//tensorflow/core/lib/io:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
        "//tensorflow/core/platform:__subpackages__",
    ],
    deps = [
        "//tensorflow/core/platform:random",
    ],
)

cc_library(
    name = "weighted_picker",
    hdrs = ["weighted_picker.h"],
    deps = [
        ":philox",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/random:weighted_picker",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "distribution_sampler.h",
        "exact_uniform_int.h",
        "philox_random.h",
        "random.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "simple_philox.h",
        "weighted_picker.h",
        "//tensorflow/tsl/lib/random:mobile_srcs_only_runtime",
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
        "//tensorflow/tsl/lib/random:legacy_lib_random_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_random_headers",
    srcs = [
        "random.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "weighted_picker.h",
        "//tensorflow/tsl/lib/random:legacy_lib_internal_public_random_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_random_all_headers",
    srcs = [
        "distribution_sampler.h",
        "exact_uniform_int.h",
        "philox_random.h",
        "random.h",
        "random_distributions.h",
        "random_distributions_utils.h",
        "simple_philox.h",
        "weighted_picker.h",
        "//tensorflow/tsl/lib/random:legacy_lib_random_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
