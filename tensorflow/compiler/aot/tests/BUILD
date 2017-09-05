licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:private"],
)

load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

test_suite(
    name = "all_tests",
    tags = ["manual"],
    tests = [
        ":test_graph_tfadd_test",
        ":test_graph_tfadd_with_ckpt_saver_test",
        ":test_graph_tfadd_with_ckpt_test",
        ":test_graph_tfgather_test",
        ":test_graph_tfmatmul_test",
        ":test_graph_tfmatmulandadd_test",
        ":tfcompile_test",
    ],
)

py_binary(
    name = "make_test_graphs",
    testonly = 1,
    srcs = ["make_test_graphs.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

genrule(
    name = "gen_test_graphs",
    testonly = 1,
    outs = [
        "test_graph_tfadd.pb",
        "test_graph_tfadd_with_ckpt.ckpt",
        "test_graph_tfadd_with_ckpt.pb",
        "test_graph_tfadd_with_ckpt_saver.ckpt",
        "test_graph_tfadd_with_ckpt_saver.pb",
        "test_graph_tfadd_with_ckpt_saver.saver",
        "test_graph_tffunction.pb",
        "test_graph_tfgather.pb",
        "test_graph_tfmatmul.pb",
        "test_graph_tfmatmulandadd.pb",
        "test_graph_tfsplits.pb",
    ],
    cmd = "$(location :make_test_graphs) --out_dir $(@D)",
    tags = ["manual"],
    tools = [":make_test_graphs"],
)

tf_library(
    name = "test_graph_tfadd",
    testonly = 1,
    config = "test_graph_tfadd.config.pbtxt",
    cpp_class = "AddComp",
    graph = "test_graph_tfadd.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt.ckpt",
    graph = "test_graph_tfadd_with_ckpt.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfadd_with_ckpt_saver",
    testonly = 1,
    config = "test_graph_tfadd_with_ckpt.config.pbtxt",
    cpp_class = "AddWithCkptSaverComp",
    freeze_checkpoint = "test_graph_tfadd_with_ckpt_saver.ckpt",
    freeze_saver = "test_graph_tfadd_with_ckpt_saver.saver",
    graph = "test_graph_tfadd_with_ckpt_saver.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfgather",
    testonly = 1,
    config = "test_graph_tfgather.config.pbtxt",
    cpp_class = "GatherComp",
    graph = "test_graph_tfgather.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfmatmul",
    testonly = 1,
    config = "test_graph_tfmatmul.config.pbtxt",
    cpp_class = "foo::bar::MatMulComp",
    graph = "test_graph_tfmatmul.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfmatmulandadd",
    testonly = 1,
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "MatMulAndAddComp",
    graph = "test_graph_tfmatmulandadd.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tffunction",
    testonly = 1,
    config = "test_graph_tffunction.config.pbtxt",
    cpp_class = "FunctionComp",
    graph = "test_graph_tffunction.pb",
    tags = ["manual"],
)

tf_library(
    name = "test_graph_tfsplits",
    testonly = 1,
    config = "test_graph_tfsplits.config.pbtxt",
    cpp_class = "SplitsComp",
    graph = "test_graph_tfsplits.pb",
    tags = ["manual"],
)

cc_test(
    name = "tfcompile_test",
    srcs = ["tfcompile_test.cc"],
    tags = ["manual"],
    deps = [
        ":test_graph_tfadd",
        ":test_graph_tfadd_with_ckpt",
        ":test_graph_tfadd_with_ckpt_saver",
        ":test_graph_tffunction",
        ":test_graph_tfgather",
        ":test_graph_tfmatmul",
        ":test_graph_tfmatmulandadd",
        ":test_graph_tfsplits",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
