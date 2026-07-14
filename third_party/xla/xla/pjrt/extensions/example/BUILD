load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# Only the plugin code should be accessing the opaque types, do not make these visible to the
# consumer of the plugin.
package_group(
    name = "extension_private",
    packages = [
        "//xla/pjrt/plugin/...",
    ],
)

cc_library(
    name = "example_extension_cpp",
    hdrs = ["example_extension_cpp.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "example_extension",
    srcs = ["example_extension.cc"],
    hdrs = ["example_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":example_extension_private",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "example_extension_private",
    hdrs = ["example_extension_private.h"],
    visibility = [":extension_private"],
    deps = [":example_extension_cpp"],
)
