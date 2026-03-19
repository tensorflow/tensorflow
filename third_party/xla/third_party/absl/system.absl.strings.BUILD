load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "strings",
    linkopts = ["-labsl_strings"],
    deps = [
        ":internal",
        "//absl/base",
        "//absl/base:throw_delegate",
        "//absl/memory",
        "//absl/numeric:bits",
        "//absl/numeric:int128",
    ],
)

cc_library(
    name = "internal",
    linkopts = ["-labsl_strings_internal"],
    deps = [
        "//absl/base:endian",
        "//absl/base:raw_logging_internal",
    ],
)

cc_library(
    name = "cord",
    linkopts = ["-labsl_cord"],
    deps = [
        ":str_format",
        "//absl/container:compressed_tuple",
        "//absl/container:fixed_array",
        "//absl/container:inlined_vector",
        "//absl/container:layout",
    ],
)

cc_library(
    name = "str_format",
    linkopts = ["-labsl_str_format_internal"],
    deps = [
        ":strings",
        "//absl/functional:function_ref",
        "//absl/numeric:representation",
        "//absl/types:optional",
        "//absl/types:span",
    ],
)
