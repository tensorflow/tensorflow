# Python bindings for the proto serialization library.

load("//xla/tsl/platform/default:build_config.bzl", "tf_py_clif_cc")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

tf_py_clif_cc(
    name = "proto_serialization",
    srcs = ["proto_serialization.clif"],
    deps = [
        "//xla/tsl/lib/strings:proto_serialization",
    ],
)
