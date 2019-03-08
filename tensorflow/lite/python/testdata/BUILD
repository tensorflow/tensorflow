licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow/lite:build_def.bzl", "tf_to_tflite")

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
        ":gather_string",
        ":permute_float",
        ":permute_uint8",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)
