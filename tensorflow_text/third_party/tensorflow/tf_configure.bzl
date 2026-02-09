"""Setup TensorFlow as external dependency.

This is used for the generation of the dynamic libraries used for custom ops.
See: http://github.com/tensorflow/custom-op
"""

load("@python//:defs.bzl", "interpreter")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

def tf_configure():
    tensorflow_annotation = """
cc_library(
    name = "tf_header_lib",
    hdrs = glob(["site-packages/tensorflow/include/**/*"]),
    strip_include_prefix="site-packages/tensorflow/include/",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = select({
        "//conditions:default": ["site-packages/tensorflow/libtensorflow_framework.so.2"],
        "@bazel_tools//src/conditions:darwin":["site-packages/tensorflow/libtensorflow_framework.2.dylib"],
        "@bazel_tools//src/conditions:darwin_x86_64": ["site-packages/tensorflow/libtensorflow_framework.2.dylib"],
    }),
    visibility = ["//visibility:public"],
)
"""
    pip_parse(
        name = "pypi",
        annotations = {
            "numpy": package_annotation(
                additive_build_content = """
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
    name = "numpy_headers",
    deps = [":numpy_headers_2", ":numpy_headers_1"],
)
""",
            ),
            "tensorflow": package_annotation(
                additive_build_content = tensorflow_annotation,
            ),
            "tf-nightly": package_annotation(
                additive_build_content = tensorflow_annotation,
            ),
        },
        python_interpreter_target = interpreter,
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
    )
