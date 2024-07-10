# Placeholder: load py_proto_library
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "toco_conversion_log_proto",
    srcs = ["toco_conversion_log.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "toco_conversion_log_proto_py",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":toco_conversion_log_proto"],
# )
# copybara:uncomment_end
