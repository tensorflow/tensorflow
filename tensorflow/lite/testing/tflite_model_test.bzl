"""Definition for tflite_model_test rule that runs a TF Lite model accuracy test.

This rule generates targets to run a diff-based model accuracy test against
synthetic, random inputs. Future work will allow injection of "golden" inputs,
as well as more robust execution on mobile devices.

Example usage:

tflite_model_test(
    name = "simple_diff_test",
    tensorflow_model_file = "//tensorflow/lite:testdata/multi_add.pb",
    input_layer = "a,b,c,d",
    input_layer_shape = "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3",
    input_layer_type = "float,float,float,float",
    output_layer = "x,y",
)
"""

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

def tflite_model_test(
        name,
        tensorflow_model_file,
        input_layer,
        input_layer_type,
        input_layer_shape,
        output_layer,
        inference_type = "float",
        extra_conversion_flags = [],
        num_runs = 20,
        tags = [],
        size = "large"):
    """Create test targets for validating TFLite model execution relative to TF.

    Args:
      name: Generated test target name. Note that multiple targets may be
          created if `delegates` are provided.
      tensorflow_model_file: The binary GraphDef proto to run the benchmark on.
      input_layer: A list of input tensors to use in the test.
      input_layer_shape: The shape of the input layer in csv format.
      input_layer_type: The data type of the input layer(s) (int, float, etc).
      output_layer: The layer that output should be read from.
      inference_type: The data type for inference and output.
      extra_conversion_flags: Extra flags to append to those used for converting
          models to the tflite format.
      num_runs: Number of synthetic test cases to run.
      tags: Extra tags to apply to the test targets.
      size: The test size to use.
    """

    conversion_flags = [
        "--input_shapes=%s" % input_layer_shape,
        "--input_arrays=%s" % input_layer,
        "--output_arrays=%s" % output_layer,
    ] + extra_conversion_flags

    tflite_model_file = make_tflite_files(
        target_name = "tflite_" + name + "_model",
        model_file = tensorflow_model_file,
        conversion_flags = conversion_flags,
        inference_type = inference_type,
    )

    diff_args = [
        # TODO(b/134772701): Find a better way to extract the absolute path from
        # a target without relying on $(location), which doesn't work with some
        # mobile test variants. For now we use $(location), but something like
        # the following is what we want for mobile tests:
        # "--tensorflow_model=%s" % tensorflow_model_file.replace("//", "").replace(":", "/"),
        # "--tflite_model=%s" % tflite_model_file.replace("//", "").replace(":", "/"),
        "--tensorflow_model=$(location %s)" % tensorflow_model_file,
        "--tflite_model=$(location %s)" % tflite_model_file,
        "--input_layer=%s" % input_layer,
        "--input_layer_type=%s" % input_layer_type,
        "--input_layer_shape=%s" % input_layer_shape,
        "--output_layer=%s" % output_layer,
        "--num_runs_per_pass=%s" % num_runs,
    ]

    tf_cc_test(
        name = name,
        size = size,
        srcs = ["//tensorflow/lite/testing:tflite_diff_example_test.cc"],
        args = diff_args,
        data = [
            tensorflow_model_file,
            tflite_model_file,
        ],
        tags = tags,
        deps = [
            "//tensorflow/lite/testing:init_tensorflow",
            "//tensorflow/lite/testing:tflite_diff_flags",
            "//tensorflow/lite/testing:tflite_diff_util",
        ],
    )

def make_tflite_files(
        target_name,
        model_file,
        conversion_flags,
        inference_type):
    """Uses TFLite to convert and input proto to tflite flatbuffer format.

    Args:
      target_name: Generated target name.
      model_file: the path to the input file.
      conversion_flags: parameters to pass to tflite for conversion.
      inference_type: The data type for inference and output.
    Returns:
      The name of the generated file.
    """
    flags = [] + conversion_flags
    if inference_type == "float":
        flags += [
            "--inference_type=FLOAT",
            "--inference_input_type=FLOAT",
        ]
    elif inference_type == "quantized":
        flags += [
            "--inference_type=QUANTIZED_UINT8",
            "--inference_input_type=QUANTIZED_UINT8",
        ]
    else:
        fail("Invalid inference type (%s). Expected 'float' or 'quantized'" % inference_type)

    srcs = [model_file]

    # Convert from Tensorflow graphdef to tflite model.
    output_file = target_name + ".fb"

    tool = "//tensorflow/lite/python:tflite_convert"
    cmd = ("$(location %s) " +
           " --graph_def_file=$(location %s)" +
           " --output_file=$(location %s)" +
           " --input_format=TENSORFLOW_GRAPHDEF" +
           " --output_format=TFLITE " +
           " --enable_v1_converter " +
           " ".join(flags)
               .replace("std_value", "std_dev_value")
               .replace("quantize_weights=true", "quantize_weights"))

    native.genrule(
        name = target_name,
        srcs = srcs,
        tags = ["manual"],
        outs = [
            output_file,
        ],
        cmd = cmd % (tool, model_file, output_file),
        tools = [tool],
        visibility = ["//tensorflow/lite/testing:__subpackages__"],
    )
    return "//%s:%s" % (native.package_name(), output_file)
