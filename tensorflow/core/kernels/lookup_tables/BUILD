# Description:
#   OpKernels and resource templates for lookup tables.

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "lookup_table_interface",
    hdrs = ["lookup_table_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "table_resource_utils",
    hdrs = ["table_resource_utils.h"],
    deps = [
        ":lookup_table_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "table_op_utils",
    hdrs = ["table_op_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/base:core_headers",
    ],
)
