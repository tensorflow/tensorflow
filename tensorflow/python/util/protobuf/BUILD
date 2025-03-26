# Tensorflow protobuf utility package

load("//tensorflow:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "tf_py_strict_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")  # @unused

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/launchpad:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tensorflow_docs:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_proto_library(
    name = "compare_test_proto",
    testonly = 1,
    srcs = ["compare_test.proto"],
)

tf_py_strict_test(
    name = "protobuf_compare_test",
    size = "small",
    srcs = ["compare_test.py"],
    main = "compare_test.py",
    tags = ["no_pip"],  # compare_test_pb2 proto is not available in pip.
    deps = [
        ":compare_test_proto_py",
        ":protobuf",
        "//tensorflow/python/platform:test",
    ],
)

filegroup(
    name = "compare_test_proto_src",
    srcs = ["compare_test.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compare_test_py_pb2",
#     testonly = 1,
#     has_services = 0,
#     deps = [":compare_test_proto"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "protobuf",
    srcs = glob(
        ["*.py"],
        exclude = ["*_test.py"],
    ),
    compatible_with = get_compatible_with_portable(),
    visibility = visibility + [
        "//tensorflow:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tensorflow_data_validation:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tfx:__subpackages__",
    ],
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        "@com_google_protobuf//:protobuf_python",
        "//tensorflow/python/util:compat",
    ],
)
