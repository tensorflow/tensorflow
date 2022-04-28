load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

package(
    default_visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "layout_proto",
    srcs = ["layout.proto"],
    cc_api_version = 2,
    create_go_proto = False,
    create_java_proto = False,
)

# copybara:comment_begin(oss only)
alias(
    name = "layout_proto_py_pb2",
    actual = ":layout_proto_py",
)
# copybara:comment_end

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "layout_proto_py_pb2",
#     api_version = 2,
#     deps = [":layout_proto"],
# )
# copybara:uncomment_end
