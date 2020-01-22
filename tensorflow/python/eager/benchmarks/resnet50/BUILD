load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "resnet50",
    srcs = ["resnet50.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "resnet50_test_util",
    srcs = ["resnet50_test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet50",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "resnet50_test_lib",
    srcs = ["resnet50_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

cuda_py_test(
    name = "resnet50_test",
    size = "medium",
    srcs = ["resnet50_test.py"],
    additional_deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/141617449): needs investigation
        "optonly",
        "oss_serial",
        "v1only",
    ],
)

cuda_py_test(
    name = "hvp_test",
    size = "medium",
    srcs = ["hvp_test.py"],
    additional_deps = [
        ":resnet50",
        ":resnet50_test_util",
        "//tensorflow/python/eager:forwardprop",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
    shard_count = 7,
    tags = [
        "no_windows",  # TODO(b/141617449): needs investigation
        "optonly",
        "oss_serial",
        "v1only",
    ],
    # Times out
    xla_enable_strict_auto_jit = False,
)

cuda_py_test(
    name = "resnet50_graph_test",
    size = "medium",
    srcs = ["resnet50_graph_test.py"],
    additional_deps = [
        ":resnet50",
        ":resnet50_test_lib",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/141617449): needs investigation
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
        "oss_serial",
    ],
)
