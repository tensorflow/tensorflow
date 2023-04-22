# Description:
# The cuDNN Frontend API is a C++ header-only library that demonstrates how
# to use the cuDNN C backend API.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "cudnn_frontend_header_files",
    srcs = glob([
        "include/**",
    ]),
)

cc_library(
    name = "cudnn_frontend",
    hdrs = [":cudnn_frontend_header_files"],
    include_prefix = "third_party/cudnn_frontend",
)
