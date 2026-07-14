# Description:
#   Doc generator

load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")
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

py_library(
    name = "tf_doctest_lib",
    srcs = ["tf_doctest_lib.py"],
    strict_deps = True,
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow_text/google:__pkg__",
    ],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tf_doctest",
    srcs = ["tf_doctest.py"],
    args = ["--module_prefix_skip=" + ",".join(tpu_module + distribute_module + keras_module)],
    shard_count = 4,
    strict_deps = True,
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

py_test(
    name = "tf_doctest_gpu",
    srcs = ["tf_doctest.py"],
    args = [
        "--module=distribute.",
        "--module_prefix_skip=" + ",".join(tpu_module),
        "--required_gpus=2",
    ],
    main = "tf_doctest.py",
    strict_deps = True,
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

py_test(
    name = "tf_doctest_test",
    srcs = ["tf_doctest_test.py"],
    strict_deps = True,
    tags = ["no_pip"],
    deps = [
        ":tf_doctest_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "fenced_doctest_lib",
    srcs = ["fenced_doctest_lib.py"],
    strict_deps = True,
    deps = [
        ":tf_doctest_lib",
        "@pypi//astor",
    ],
)

py_test(
    name = "fenced_doctest_test",
    srcs = ["fenced_doctest_test.py"],
    strict_deps = True,
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

py_library(
    name = "doc_controls",
    srcs = ["doc_controls.py"],
    compatible_with = get_compatible_with_portable(),
    strict_deps = True,
    visibility = ["//visibility:public"],
)

py_test(
    name = "generate2_test",
    size = "medium",
    srcs = ["generate2_test.py"],
    strict_deps = True,
    tags = [
        "manual",
        "optonly",
    ],
    deps = [
        ":generate2_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:test",
        "@pypi//packaging",
    ],
)

py_binary(
    name = "generate2",
    srcs = ["generate2.py"],
    strict_deps = True,
    deps = [
        ":generate2_lib",
    ],
)

py_library(
    # Opensource only
    name = "base_dir_oss",
    srcs = ["base_dir.py"],
    strict_deps = False,
    deps = [],
)

py_library(
    name = "generate2_lib",
    srcs = ["generate2.py"],
    strict_deps = True,
    deps = [
        ":base_dir_oss",
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:pywrap_xla_ops",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@pypi//packaging",
    ],
)

py_binary(
    name = "build_cc_api_headers",
    srcs = ["build_cc_api_headers.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    strict_deps = True,
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
