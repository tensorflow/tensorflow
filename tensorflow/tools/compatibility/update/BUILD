load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "generate_v2_renames_map",
    srcs = ["generate_v2_renames_map.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:modules_with_exports",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:all_renames_v2",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "generate_v2_reorders_map",
    srcs = ["generate_v2_reorders_map.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:no_contrib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "//tensorflow/tools/compatibility:tf_upgrade_v2_lib",
        "@absl_py//absl:app",
    ],
)
