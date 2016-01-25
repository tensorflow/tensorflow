# Description:
#   Doc generator

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//tensorflow:__subpackages__"],
)

py_binary(
    name = "gen_cc_md",
    srcs = ["gen_cc_md.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

filegroup(
    name = "doxy_config",
    srcs = ["tf-doxy_for_md-config"],
)

sh_binary(
    name = "gen_docs",
    srcs = ["gen_docs.sh"],
    data = [
        ":doxy_config",
        ":gen_cc_md",
        "//tensorflow/python:gen_docs_combined",
    ],
)

sh_test(
    name = "gen_docs_test",
    size = "small",
    srcs = [
        "gen_docs_test.sh",
    ],
    data = [
        ":gen_docs",
        "//tensorflow/core:all_files",
        "//tensorflow/python:all_files",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
