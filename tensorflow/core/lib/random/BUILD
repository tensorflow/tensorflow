load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
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

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

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
    deps = [
        ":exact_uniform_int",
        ":philox_random",
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "philox_random",
    hdrs = ["philox_random.h"],
    deps = ["//tensorflow/core/platform:types"],
)

cc_library(
    name = "philox_random_test_utils",
    testonly = True,
    hdrs = ["philox_random_test_utils.h"],
    deps = [
        ":philox_random",
        ":random",
        "//tensorflow/core/platform:logging",
    ],
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
    srcs = ["weighted_picker.cc"],
    hdrs = ["weighted_picker.h"],
    deps = [
        ":philox",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
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
        "random.h",
        "random_distributions.h",
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
        "simple_philox.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_random_headers",
    srcs = [
        "random.h",
        "random_distributions.h",
        "weighted_picker.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "philox_random_test_utils.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_random_all_headers",
    srcs = [
        "distribution_sampler.h",
        "exact_uniform_int.h",
        "philox_random.h",
        "philox_random_test_utils.h",
        "random.h",
        "random_distributions.h",
        "simple_philox.h",
        "weighted_picker.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_random_tests",
    srcs = [
        "distribution_sampler_test.cc",
        "philox_random_test.cc",
        "random_distributions_test.cc",
        "random_test.cc",
        "simple_philox_test.cc",
        "weighted_picker_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_random_random_distributions_test",
    srcs = [
        "random_distributions_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_random_random_weighted_picker_test",
    srcs = [
        "weighted_picker_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
