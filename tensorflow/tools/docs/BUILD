# Description:
#   Doc generator

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

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
    srcs_version = "PY3",
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
    python_version = "PY3",
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
        "//tensorflow/python:logging_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_test(
    name = "tf_doctest_tpu",
    srcs = ["tf_doctest.py"],
    args = ["--module=" + ",".join(tpu_module)],
    disable_experimental = True,
    disable_tfrt = True,
    disable_v3 = True,
    main = "tf_doctest.py",
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
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
    python_version = "PY3",
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
        "//tensorflow/python:logging_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "tf_doctest_test",
    srcs = ["tf_doctest_test.py"],
    python_version = "PY3",
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
    deps = [
        ":tf_doctest_lib",
        "@astor_archive//:astor",
    ],
)

py_test(
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

py_library(
    name = "doc_controls",
    srcs = ["doc_controls.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "generate2_test",
    size = "medium",
    srcs = ["generate2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "optonly",
    ],
    deps = [
        ":generate2_lib",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_estimator",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "generate2",
    srcs = ["generate2.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":generate2_lib",
    ],
)

py_library(
    # Opensource only
    name = "base_dir_oss",
    srcs = ["base_dir.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "generate2_lib",
    srcs = ["generate2.py"],
    srcs_version = "PY3",
    deps = [
        ":base_dir_oss",
        # copybara:uncomment "//third_party/py/tensorflow:tensorflow_estimator",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "build_cc_api_headers",
    srcs = ["build_cc_api_headers.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "build_java_api_docs",
    srcs = ["build_java_api_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
