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
    deps = [":data_type"],
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

cc_library(
    name = "data_type",
    hdrs = ["data_type.h"],
    deps = [
        ":bf16",
        ":f16",
    ],
)

cc_library(
    name = "dispatch",
    hdrs = ["dispatch.h"],
    visibility = ["//visibility:private"],
    deps = [":data_type"],
)
