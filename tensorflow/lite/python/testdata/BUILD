load("//tensorflow/lite:build_def.bzl", "tf_to_tflite")
load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0,
)

exports_files(glob(["*.pb"]))

tf_to_tflite(
    name = "permute_float",
    src = "permute.pbtxt",
    out = "permute_float.tflite",
    options = [
        "--input_arrays=input",
        "--output_arrays=output",
    ],
)

tf_to_tflite(
    name = "permute_uint8",
    src = "permute.pbtxt",
    out = "permute_uint8.tflite",
    options = [
        "--input_arrays=input",
        "--output_arrays=output",
        "--inference_type=QUANTIZED_UINT8",
        "--std_values=1",
        "--mean_values=0",
        "--default_ranges_min=0",
        "--default_ranges_max=255",
    ],
)

tf_to_tflite(
    name = "gather_string",
    src = "gather.pbtxt",
    out = "gather_string.tflite",
    options = [
        "--input_arrays=input,indices",
        "--output_arrays=output",
    ],
)

filegroup(
    name = "interpreter_test_data",
    srcs = [
        "pc_conv.bin",
        ":gather_string",
        ":permute_float",
        ":permute_uint8",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "test_delegate",
    testonly = 1,
    srcs = ["test_delegate.cc"],
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/lite/c:common",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "test_delegate.so",
    testonly = 1,
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":test_delegate",
    ],
)

cc_library(
    name = "test_registerer",
    srcs = ["test_registerer.cc"],
    hdrs = ["test_registerer.h"],
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

pybind_extension(
    name = "_pywrap_test_registerer",
    srcs = [
        "test_registerer_wrapper.cc",
    ],
    hdrs = ["test_registerer.h"],
    additional_exported_symbols = ["TF_TestRegisterer"],
    link_in_framework = True,
    module_name = "_pywrap_test_registerer",
    deps = [
        ":test_registerer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
