load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "any",
    deps = [
        ":bad_any_cast",
    ],
)

cc_library(
    name = "bad_any_cast",
    linkopts = ["-labsl_bad_any_cast_impl"],
)

cc_library(
    name = "span",
    deps = [
        "//absl/base:throw_delegate",
    ],
)

cc_library(
    name = "optional",
    deps = [
        ":bad_optional_access",
    ],
)

cc_library(
    name = "bad_optional_access",
    linkopts = ["-labsl_bad_optional_access"],
    deps = [
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "bad_variant_access",
    linkopts = ["-labsl_bad_variant_access"],
    deps = [
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "variant",
    deps = [
        ":bad_variant_access",
    ],
)

cc_library(
    name = "compare",
    deps = [
        "//absl/meta:type_traits",
    ],
)
