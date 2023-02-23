# Libraries to support TfLite buffer / synchronization interoperability.
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_test(
    name = "reconcile_fns_test",
    srcs = ["reconcile_fns_test.cc"],
    deps = [
        ":attribute_keys",
        ":attribute_map_internal",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "attribute_keys",
    hdrs = ["attribute_keys.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "variant",
    srcs = ["variant.cc"],
    hdrs = ["variant.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_test(
    name = "variant_test",
    srcs = ["variant_test.cc"],
    deps = [
        ":variant",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "attribute_map_internal_test",
    srcs = ["attribute_map_internal_test.cc"],
    deps = [
        ":attribute_keys",
        ":attribute_map_internal",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "attribute_map_internal",
    srcs = [
        "attribute_map_internal.cc",
        "reconcile_fns.cc",
    ],
    hdrs = [
        "attribute_map_internal.h",
        "reconcile_fns.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":attribute_keys",
        ":variant",
        "//tensorflow/lite/core/async/interop/c:types",
    ],
)
