# Description:
#   Doc generator

load("//tensorflow:py.default.bzl", "py_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],
)

tpu_module = [
    "tpu.",
    "distribute.tpu_strategy",
    "distribute.cluster_resolver.tpu",
    "distribute.cluster_resolver.tpu_oss",
]

keras_module = [
    "keras.",
]

# tf.distribute docstring often uses GPU, so they're only covered in
# tf_doctest_gpu.
distribute_module = [
    "distribute.",
]

py_strict_library(
    name = "tf_doctest_lib",
    srcs = ["tf_doctest_lib.py"],
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow_text/google:__pkg__",
    ],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tf_doctest",
    srcs = ["tf_doctest.py"],
    args = ["--module_prefix_skip=" + ",".join(tpu_module + distribute_module + keras_module)],
    shard_count = 4,
    tags = [
        "no_oss",  # b/275546007
        "no_pip",
        "no_rocm",  # No need to rerun this test for ROCm config.
        "no_windows",  # numpy prints differently on windows.
        "noasan",
    ],
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:logging_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_strict_test(
    name = "tf_doctest_tpu",
    srcs = ["tf_doctest.py"],
    args = ["--module=" + ",".join(tpu_module)],
    disable_experimental = True,
    disable_tfrt = True,
    disable_v3 = True,
    main = "tf_doctest.py",
    tags = ["no_oss"],
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:logging_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "tf_doctest_gpu",
    srcs = ["tf_doctest.py"],
    args = [
        "--module=distribute.",
        "--module_prefix_skip=" + ",".join(tpu_module),
        "--required_gpus=2",
    ],
    main = "tf_doctest.py",
    tags = [
        "no_pip",
        "no_rocm",
        "no_windows",  # numpy prints differently on windows.
        "noasan",
        "nomsan",
        "notsan",
    ] + tf_gpu_tests_tags(),
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:logging_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "tf_doctest_test",
    srcs = ["tf_doctest_test.py"],
    tags = ["no_pip"],
    deps = [
        ":tf_doctest_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "fenced_doctest_lib",
    srcs = ["fenced_doctest_lib.py"],
    deps = [
        ":tf_doctest_lib",
        "@pypi_astor//:pkg",
    ],
)

py_strict_test(
    name = "fenced_doctest_test",
    srcs = ["fenced_doctest_test.py"],
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":fenced_doctest_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "doc_controls",
    srcs = ["doc_controls.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

py_strict_test(
    name = "generate2_test",
    size = "medium",
    srcs = ["generate2_test.py"],
    tags = [
        "manual",
        "optonly",
    ],
    deps = [
        ":generate2_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:test",
        "@pypi_packaging//:pkg",
    ],
)

py_strict_binary(
    name = "generate2",
    srcs = ["generate2.py"],
    deps = [
        ":generate2_lib",
    ],
)

py_library(
    # Opensource only
    name = "base_dir_oss",
    srcs = ["base_dir.py"],
    deps = [],
)

py_strict_library(
    name = "generate2_lib",
    srcs = ["generate2.py"],
    deps = [
        ":base_dir_oss",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:pywrap_xla_ops",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@pypi_packaging//:pkg",
    ],
)

py_strict_binary(
    name = "build_cc_api_headers",
    srcs = ["build_cc_api_headers.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
