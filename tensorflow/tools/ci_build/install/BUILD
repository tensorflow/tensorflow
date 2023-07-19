"""Description: BUILD file for shell script to install bazel using the installer.
"""

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = glob(["*.sh"]),
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "install_bazel",
    srcs = ["install_bazel.sh"],
)

sh_binary(
    name = "install_bazel_from_source",
    srcs = ["install_bazel_from_source.sh"],
)
