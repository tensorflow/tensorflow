# copybara:uncomment load("@rules_license//rules:license.bzl", "license")

package(
    # copybara:uncomment_begin
    # default_applicable_licenses = ["@triton//:license"],
    # default_compatible_with = ["//buildenv/target:non_prod"],
    # default_visibility = [
    # "@triton//:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = ["//visibility:public"],
    # copybara:comment_end
)

# copybara:uncomment_begin
# license(
#     name = "license",
#     license_text = "LICENCE.txt",
# )
#
# licenses(["notice"])
#
# exports_files(["LICENCE.txt"])
# copybara:uncomment_end

cc_library(
    name = "f2reduce",
    srcs = ["f2reduce.cpp"],
    hdrs = ["f2reduce.h"],
    # copybara:uncomment strip_include_prefix = "/third_party/triton",
)
