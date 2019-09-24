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
    name = "resnet50_test_lib",
    srcs = ["resnet50_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet50",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

cuda_py_test(
    name = "resnet50_test",
    size = "medium",
    srcs = ["resnet50_test.py"],
    additional_deps = [
        ":resnet50",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",  # needs investigation
        "optonly",
        "oss_serial",
    ],
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
        "no_windows",  # needs investigation
        "noasan",
        "nomsan",
        "notsan",
        "optonly",
        "oss_serial",
    ],
)
