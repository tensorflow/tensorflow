load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/lite:build_def.bzl",
    "common_test_args_for_generated_models",
    "common_test_tags_for_generated_models",
    "generated_test_models",
    "max_number_of_test_models_in_merged_zip",
    "merged_test_model_name",
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
        "avg_pool3d",
        "broadcast_args",
        "broadcast_to",
        "broadcast_gradient_args",
        "cond",
        "complex_abs",
        "control_dep",
        "conv_bias_relu6",
        "conv3d",
        "conv3d_transpose",
        "cumsum",
        # copybara:uncomment_begin(Exclude tests that depend on tensorflow_addons APIs)
        # "dense_image_warp",
        # copybara:uncomment_end
        "dynamic_rnn",
        "einsum",
        "expm1",
        "identify_dilated_conv",
        "identify_dilated_conv1d",
        "imag",
        "irfft2d",
        "is_finite",
        "max_pool3d",
        "max_pool_with_argmax",
        "parse_example",
        "random_standard_normal",
        "random_uniform",
        "real",
        "reciprocal",
        "reduce_all",
        "rfft",
        "rfft2d",
        "roll",
        "roll_with_constant",
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
        "tensor_scatter_add",
        "tensor_scatter_update",
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

# The following test gen helpers are branched from tensorflow/lite/build_def.bzl
# TODO(b/192473002): Clean up duplicated test fixtures once toco converter
# zip tests are cleaned up.
def generated_test_models_successful(conversion_mode):
    """Returns the list of successful test models.

    Args:
      conversion_mode: Conversion mode.

    Returns:
      List of successful test models for the conversion mode.
    """
    return [test_model for test_model in generated_test_models() if test_model not in generated_test_models_failing(conversion_mode)]

def number_of_merged_zip_file(conversion_mode):
    """Returns the number of merged zip file targets.

    Returns:
      Number of merged zip file targets.
    """
    m = max_number_of_test_models_in_merged_zip()
    return (len(generated_test_models_successful(conversion_mode)) + m - 1) // m

def merged_test_models():
    """Generates a list of merged tests with the different converters.

    This model list should be referred only if :generate_examples supports
    --no_tests_limit and --test_sets flags.

    Returns:
      List of tuples representing:
            (conversion mode, name of group, test tags, test args).
    """
    conversion_modes = generated_test_conversion_modes()
    options = []
    for conversion_mode in conversion_modes:
        test = merged_test_model_name()
        if conversion_mode:
            test += "_%s" % conversion_mode
        successful_tests = generated_test_models_successful(conversion_mode)
        if len(successful_tests) > 0:
            tags = common_test_tags_for_generated_models(conversion_mode, False)

            # Only non-merged tests are executed on TAP.
            # Merged test rules are only for running on the real device environment.
            if "notap" not in tags:
                tags.append("notap")
            args = common_test_args_for_generated_models(conversion_mode, False)
            n = number_of_merged_zip_file(conversion_mode)
            for i in range(n):
                test_i = "%s_%d" % (test, i)
                options.append((conversion_mode, test_i, tags, args))
    return options

def flags_for_merged_test_models(test_name, conversion_mode):
    """Returns flags for generating zipped-example data file for merged tests.

    Args:
      test_name: str. Test name in the form of "<merged_model_name>_[<conversion_mode>_]%d".
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.

    Returns:
      Flags for generating zipped-example data file for merged tests.
    """
    prefix = merged_test_model_name() + "_"
    if not test_name.startswith(prefix):
        fail(msg = "Invalid test name " + test_name + ": test name should start " +
                   "with " + prefix + " when using flags of merged test models.")

    # Remove prefix and conversion_mode from the test name
    # to extract merged test index number.
    index_string = test_name[len(prefix):]
    if conversion_mode:
        index_string = index_string.replace("%s_" % conversion_mode, "")

    # If the maximum number of test models in a file is 15 and the number of
    # successful test models are 62, 5 zip files will be generated.
    # To assign the test models fairly among these files, each zip file
    # should contain 12 or 13 test models. (62 / 5 = 12 ... 2)
    # Each zip file will have 12 test models and the first 2 zip files will have
    # 1 more test model each, resulting [13, 13, 12, 12, 12] assignment.
    # So Zip file 0, 1, 2, 3, 4 and 5 will have model[0:13], model[13:26],
    # model[26,38], model[38,50] and model[50,62], respectively.
    zip_index = int(index_string)
    num_merged_zips = number_of_merged_zip_file(conversion_mode)
    test_models = generated_test_models_successful(conversion_mode)

    # Each zip file has (models_per_zip) or (models_per_zip+1) test models.
    models_per_zip = len(test_models) // num_merged_zips

    # First (models_remaining) zip files have (models_per_zip+1) test models each.
    models_remaining = len(test_models) % num_merged_zips
    if zip_index < models_remaining:
        # Zip files [0:models_remaining] have (models_per_zip+1) models.
        begin = (models_per_zip + 1) * zip_index
        end = begin + (models_per_zip + 1)
    else:
        # Zip files [models_remaining:] have (models_per_zip) models.
        begin = models_per_zip * zip_index + models_remaining
        end = begin + models_per_zip
    tests_csv = ""
    for test_model in test_models[begin:end]:
        tests_csv += "%s," % test_model
    if tests_csv != "":
        tests_csv = tests_csv[:-1]  # Remove trailing comma.
    return " --no_tests_limit --test_sets=%s" % tests_csv

def mlir_generated_test_models():
    """Returns a list of models to be tested with MLIR-based conversion.

    Returns:
      List of strings of models.
    """
    models = []
    denylisted_models = mlir_generated_test_denylisted_models()
    for model in generated_test_models() + mlir_only_generated_test_models():
        if model not in denylisted_models:
            models.append(model)
    return models

def generated_test_conversion_modes():
    """Returns a list of conversion modes."""

    # TODO(b/146025965): Remove reference to toco.
    return ["toco-flex", "forward-compat", "", "mlir-quant"]

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

def gen_zip_test(
        name,
        test_name,
        conversion_mode,
        tags,
        args,
        additional_test_tags_args = {},
        **kwargs):
    """Generate a zipped-example test and its dependent zip files.

    Args:
      name: str. Resulting cc_test target name
      test_name: str. Test targets this model. Comes from the list above.
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.
      tags: tags for the generated cc_test.
      args: the basic cc_test args to be used.
      additional_test_tags_args: a dictionary of additional test tags and args
        to be used together with test_tags and test_args. The key is an
        identifier which can be in creating a test tag to identify a set of
        tests. The value is a tuple of list of additional test tags and args to
        be used.
      **kwargs: tf_cc_test kwargs
    """
    flags = ""

    if conversion_mode == "forward-compat":
        flags += " --make_forward_compat_test"
    elif conversion_mode == "mlir-quant":
        flags += " --mlir_quantizer"
        # TODO(b/146025965): Remove reference to toco.

    elif conversion_mode == "toco-flex":
        flags += " --ignore_converter_errors --run_with_flex"
    if test_name.startswith(merged_test_model_name() + "_"):
        flags += flags_for_merged_test_models(test_name, conversion_mode)
    gen_zipped_test_file(
        name = "zip_%s" % test_name,
        file = "%s.zip" % test_name,
        flags = flags,
    )
    tf_cc_test(name, tags = tags, args = args, **kwargs)

    for key, value in additional_test_tags_args.items():
        additional_tags, additional_args = value
        additional_tags.append("gen_zip_test_%s" % key)
        tf_cc_test(
            name = "%s_%s" % (name, key),
            args = args + additional_args,
            tags = tags + additional_tags,
            **kwargs
        )

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
