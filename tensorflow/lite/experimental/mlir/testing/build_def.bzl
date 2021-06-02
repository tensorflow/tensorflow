load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/lite:build_def.bzl",
    "generated_test_models",
)

# This is forked from `tensorflow/lite/build_def.bzl`.
# TODO(b/136499575): Merge this back to TFLite codebase when open sourcing.
def mlir_generated_test_denylisted_models():
    return [
        # TODO(b/150647400): This test passes in TF2 with tf.compat.v1 but
        # fails in TF1 with tf.compat.v1. Due to the testing environments
        # changing on 3/3, this will only be disabled temporarily.
        "unidirectional_sequence_lstm",
        "unidirectional_sequence_rnn",
    ]

# Test cases which only work with MLIR-based conversion now.
def mlir_only_generated_test_models():
    return [
        "batchmatmul",
        "broadcast_to",
        "broadcast_gradient_args",
        "cond",
        "complex_abs",
        "control_dep",
        "conv_bias_relu6",
        "conv3d",
        "cumsum",
        # TODO(b/186563810): Enable after resolving tensorflow_addons dep issue
        # that causes test failures in the exported codebase.
        # copybara:uncomment_begin
        # "dense_image_warp",
        # copybara:uncomment_end
        "dynamic_rnn",
        "einsum",
        "identify_dilated_conv",
        "identify_dilated_conv1d",
        "imag",
        "irfft2d",
        "is_finite",
        "max_pool_with_argmax",
        "parse_example",
        "real",
        "reciprocal",
        "reduce_all",
        "rfft",
        "rfft2d",
        "segment_sum",
        "shape_to_strided_slice",
        "softplus",
        "static_hashtable",
        "static_rnn_with_control_flow_v2",
        "stft",
        "tensor_list_concat",
        "tensor_list_get_item",
        "tensor_list_length",
        "tensor_list_resize",
        "tensor_list_set_item",
        "tensor_list_dynamic_shape",
        "where_v2",
        "while",
    ]

# Test cases which only work internally now.
def no_oss_generated_test_models():
    return [
        "sparse_to_dense",
    ]

# List of models that fail generated tests for the conversion mode.
# If you have to disable a test, please add here with a link to the appropriate
# bug or issue.
def generated_test_models_failing(conversion_mode):
    return []

def mlir_generated_test_models():
    """Returns a list of models to be tested with MLIR-based conversion."""
    models = []
    denylisted_models = mlir_generated_test_denylisted_models()
    for model in generated_test_models() + mlir_only_generated_test_models():
        if model not in denylisted_models:
            models.append(model)
    return models

def generated_test_conversion_modes():
    """Returns a list of conversion modes."""

    return ["forward-compat", "", "mlir-quant"]

def generated_test_models_all():
    """Generates a list of all tests with the different converters.

    Returns:
      List of tuples representing:
            (conversion mode, name of test, test tags, test args).
    """
    conversion_modes = generated_test_conversion_modes()
    no_oss_tests = no_oss_generated_test_models()
    options = []
    for conversion_mode in conversion_modes:
        failing_tests = generated_test_models_failing(conversion_mode)
        for test in mlir_generated_test_models():
            tags = []
            args = []

            # TODO(b/187992093): Exclude tests that are failing in OSS for now.
            if test in no_oss_tests:
                tags.append("no_oss")

            # Forward-compat coverage testing is largely redundant, and
            # contributes to coverage test bloat.
            if conversion_mode == "forward-compat":
                tags.append("nozapfhahn")

            if test in failing_tests:
                tags.append("notap")
                tags.append("manual")
            if conversion_mode:
                test += "_%s" % conversion_mode
            options.append((conversion_mode, test, tags, args))

    return options

def gen_zip_test(name, test_name, conversion_mode, **kwargs):
    """Generate a zipped-example test and its dependent zip files.

    Args:
      name: str. Resulting cc_test target name
      test_name: str. Test targets this model. Comes from the list above.
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.
      **kwargs: tf_cc_test kwargs
    """
    flags = ""

    if conversion_mode == "forward-compat":
        flags += " --make_forward_compat_test"
    elif conversion_mode == "mlir-quant":
        flags += " --mlir_quantizer"

    gen_zipped_test_file(
        name = "zip_%s" % test_name,
        file = "%s.zip" % test_name,
        flags = flags,
    )
    tf_cc_test(name, **kwargs)

def gen_zipped_test_file(name, file, flags = ""):
    """Generate a zip file of tests by using :generate_examples.

    Args:
      name: str. Name of output. We will produce "`file`.files" as a target.
      file: str. The name of one of the generated_examples targets, e.g. "transpose"
      flags: str. Any additional flags to include
    """
    native.genrule(
        name = file + ".files",
        cmd = (("$(locations :generate_examples) " +
                " --zip_to_output {0} {1} $(@D)").format(file, flags)),
        outs = [file],
        # `exec_tools` is required for PY3 compatibility in place of `tools`.
        exec_tools = [
            ":generate_examples",
        ],
    )

    native.filegroup(
        name = name,
        srcs = [file],
    )
