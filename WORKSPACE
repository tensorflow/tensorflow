def tf_workspace1():
    load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

    # TensorFlow Workspace 1
    http_archive(
        name = "org_tensorflow",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/v2.7.0.tar.gz",
        ],
        sha256 = "fd30d5a60f8a0973ef4b0b957e07889102ed44f4b1726a444db01b19b44883e9",
        strip_prefix = "tensorflow-2.7.0",
        build_file_content = """
py_library(
    name = "tensorflow",
    srcs = glob(["tensorflow/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [],
)
""",
    )
    load("@org_tensorflow//:tensorflow.bzl", "tf_workspace1")
    tf_workspace1()

def tf_workspace0():
    load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

    # TensorFlow Workspace 0
    http_archive(
        name = "org_tensorflow",
        urls = [
            "https://github.com/tensorflow/tensorflow/archive/v2.7.0.tar.gz",
        ],
        sha256 = "fd30d5a60f8a0973ef4b0b957e07889102ed44f4b1726a444db01b19b44883e9",
        strip_prefix = "tensorflow-2.7.0",
        build_file_content = """
py_library(
    name = "tensorflow",
    srcs = glob(["tensorflow/**/*.py"]),
    visibility = ["//visibility:public"],
    deps = [],
)
""",
    )
    load("@org_tensorflow//:tensorflow.bzl", "tf_workspace0")
    tf_workspace0()

tf_workspace0()

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/f7388651ee99608fb5f6336764657596e2f84b97.tar.gz",
    strip_prefix = "bazel-compile-commands-extractor-f7388651ee99608fb5f6336764657596e2f84b97",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
hedron_compile_commands_setup()
