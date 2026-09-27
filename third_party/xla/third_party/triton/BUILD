# copybara:uncomment package(default_applicable_licenses = ["//third_party/tensorflow:license"])

filegroup(
    name = "patch_files",
    srcs = glob([
        "common/**",
        "oss_only/**",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "workspace",
    srcs = ["workspace.bzl"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
)
