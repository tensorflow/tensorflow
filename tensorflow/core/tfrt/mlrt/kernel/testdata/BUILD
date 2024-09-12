load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/tfrt/mlrt/kernel:__subpackages__",
    ],
)

filegroup(
    name = "testdata",
    srcs = [
        "gen_checkpoint_data/saved_model.pb",
        "gen_checkpoint_data/variables/variables.data-00000-of-00001",
        "gen_checkpoint_data/variables/variables.index",
    ],
    visibility = ["//visibility:public"],
)

py_strict_binary(
    name = "gen_checkpoint",
    srcs = ["gen_checkpoint.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

genrule(
    name = "checkpoint_data",
    srcs = [],
    outs = [
        "gen_checkpoint_data/variables/variables.data-00000-of-00001",
        "gen_checkpoint_data/variables/variables.index",
        "gen_checkpoint_data/saved_model.pb",
    ],
    cmd = "$(location gen_checkpoint)" + " --saved_model_path=$(RULEDIR)/gen_checkpoint_data",
    tools = ["gen_checkpoint"],
)
