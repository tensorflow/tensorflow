load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

cc_library(
    name = "example_extension_cpp",
    srcs = ["example_extension_cpp.cc"],
    hdrs = ["example_extension_cpp.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
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
        ":example_extension_cpp",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)
