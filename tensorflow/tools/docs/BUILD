# Description:
#   Doc generator

load("//tensorflow:tensorflow.bzl", "py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

tpu_module = [
    "tpu.",
    "distribute.tpu_strategy",
    "distribute.cluster_resolver.tpu",
    "distribute.cluster_resolver.tpu_oss",
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
    deps = [
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tf_doctest",
    srcs = ["tf_doctest.py"],
    args = ["--module_prefix_skip=" + ",".join(tpu_module + distribute_module)],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
        "no_rocm",  # No need to rerun this test for ROCm config.
        "no_tfrt",  # TODO(b/179308349)
        "no_windows",  # numpy prints differently on windows.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tf_doctest_lib",
        "//tensorflow:tensorflow_py",
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
    disable_v3 = True,
    main = "tf_doctest.py",
    python_version = "PY3",
    tags = [
        "no_oss",
        "noasan",
        "nomsan",
        "notsan",
    ],
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
        "no_oss_py2",
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
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

py_test(
    name = "tf_doctest_test",
    srcs = ["tf_doctest_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tf_doctest_lib",
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
        # No reason to run sanitizers or fastbuild for this test.
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
    ],
    deps = [
        ":generate2_lib",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
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
