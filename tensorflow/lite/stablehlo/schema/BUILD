load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

flatbuffer_cc_library(
    name = "schema_fbs",
    srcs = ["schema.fbs"],
    compatible_with = get_compatible_with_portable(),
)
