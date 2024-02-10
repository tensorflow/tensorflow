# StableHLO Reference Library

package(
    default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "shlo",
    deps = [
        ":tensor",
    ],
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
)

cc_library(
    name = "bf16",
    hdrs = ["bf16.h"],
    deps = [":has_keyword"],
)

cc_library(
    name = "f16",
    hdrs = ["f16.h"],
    deps = [":has_keyword"],
)

cc_library(
    name = "has_keyword",
    hdrs = ["has_keyword.h"],
)
