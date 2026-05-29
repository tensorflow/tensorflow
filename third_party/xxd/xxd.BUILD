load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

exports_files(["LICENSE"])

license(
    name = "license",
    package_name = "vim",
    license_kinds = [
        "@rules_license//licenses/spdx:Vim",
    ],
    license_text = "LICENSE",
    package_url = "https://github.com/vim/vim",
)

cc_binary(
    name = "xxd",
    srcs = [
        "xxd.c",
    ],
    local_defines = select({
        "@platforms//os:windows": ["WIN32=1"],
        "//conditions:default": [],
    }),
)
