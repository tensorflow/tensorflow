# XLA SDY C APIs.
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

SDY_CAPI_SOURCES = [
    "passes.cc",
]

SDY_CAPI_HEADERS = [
    "passes.h",
]

cc_library(
    name = "xla_sdy_capi",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Header-only target, used when using the C API from a separate shared library.
cc_library(
    name = "xla_sdy_capi_headers",
    hdrs = SDY_CAPI_HEADERS,
    includes = ["."],
    deps = [
        "@llvm-project//mlir:CAPIIRHeaders",
    ],
)

# Alwayslink target, used when exporting the C API from a shared library.
cc_library(
    name = "xla_sdy_capi_objects",
    srcs = SDY_CAPI_SOURCES,
    hdrs = SDY_CAPI_HEADERS,
    deps = [
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRObjects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = True,
)
