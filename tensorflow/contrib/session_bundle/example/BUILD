# Description: TensorFlow Serving session_bundle example.

package(
    default_visibility = ["//tensorflow/contrib/session_bundle:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# vardef("PYTHON_BIN_PATH", "/usr/bin/python")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "export_half_plus_two",
    srcs = [
        "export_half_plus_two.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/session_bundle:exporter",
    ],
)

genrule(
    name = "half_plus_two",
    outs = [
        "half_plus_two/00000123/export.meta",
        "half_plus_two/00000123/export-00000-of-00001",
    ],
    cmd =
        "rm -rf /tmp/half_plus_two; " +
        "$(PYTHON_BIN_PATH) $(locations :export_half_plus_two); " +
        "cp -r /tmp/half_plus_two/* $(@D)/half_plus_two",
    tools = [
        ":export_half_plus_two",
    ],
    visibility = ["//visibility:public"],
)
