"""Build rule definitions for TFLite zip tests."""

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

# This is the master list of generated examples that will be made into tests. A
# function called make_XXX_tests() must also appear in generate_examples.py.
# Disable a test by adding it to the denylists specified in
# generated_test_models_failing().
def generated_test_models():
    return [
        # keep sorted
        "abs",
        "add",
        "add_n",
        "arg_min_max",
        "avg_pool",
        "avg_pool3d",
        "batch_to_space_nd",
        "broadcast_args",
        "broadcast_gradient_args",
        "broadcast_to",
        "cast",
        "ceil",
        "complex_abs",
        "concat",
        "cond",
        "constant",
        "control_dep",
        "conv",
        "conv2d_transpose",
        "conv3d",
        "conv3d_transpose",
        "conv_bias_relu6",
        "conv_relu",
        "conv_relu1",
        "conv_relu6",
        "conv_to_depthwiseconv_with_shared_weights",
        "conv_with_shared_weights",
        "cos",
        "cumsum",
        # copybara:uncomment(Exclude tests that depend on tensorflow_addons APIs) "dense_image_warp",
        "depth_to_space",
        "depthwiseconv",
        "div",
        "dynamic_rnn",
        "dynamic_update_slice",
        "einsum",
        "elu",
        "embedding_lookup",
        "equal",
        "exp",
        "expand_dims",
        "expm1",
        "eye",
        "fill",
        "floor",
        "floor_div",
        "floor_mod",
        "fully_connected",
        "fused_batch_norm",
        "gather",
        "gather_nd",
        "gather_with_constant",
        "gelu",
        "global_batch_norm",
        "greater",
        "greater_equal",
        "hardswish",
        "identify_dilated_conv",
        "identify_dilated_conv1d",
        "identity",
        "imag",
        "irfft2d",
        "is_finite",
        "l2_pool",
        "l2norm",
        "l2norm_shared_epsilon",
        "leaky_relu",
        "less",
        "less_equal",
        "local_response_norm",
        "log",
        "log_softmax",
        "logical_and",
        "logical_or",
        "logical_xor",
        "lstm",
        "matrix_diag",
        "matrix_set_diag",
        "max_pool",
        "max_pool3d",
        "max_pool_with_argmax",
        "maximum",
        "mean",
        "minimum",
        "mirror_pad",
        "mul",
        "multinomial",
        "nearest_upsample",
        "neg",
        "not_equal",
        "one_hot",
        "pack",
        "pad",
        "padv2",
        "parse_example",
        "placeholder_with_default",
        "pow",
        "prelu",
        "random_standard_normal",
        "random_uniform",
        "range",
        "rank",
        "real",
        "reciprocal",
        "reduce_all",
        "reduce_any",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "relu",
        "relu1",
        "relu6",
        "reshape",
        "resize_bilinear",
        "resize_nearest_neighbor",
        "resolve_constant_strided_slice",
        "reverse_sequence",
        "reverse_v2",
        "rfft",
        "rfft2d",
        "roll",
        "roll_with_constant",
        "round",
        "rsqrt",
        "scatter_nd",
        "segment_sum",
        "shape",
        "shape_to_strided_slice",
        "sigmoid",
        "sin",
        "slice",
        "softmax",
        "softplus",
        "space_to_batch_nd",
        "space_to_depth",
        "sparse_to_dense",
        "split",
        "splitv",
        "sqrt",
        "square",
        "squared_difference",
        "squeeze",
        "static_hashtable",
        "static_rnn_with_control_flow_v2",
        "stft",
        "strided_slice",
        "strided_slice_1d_exhaustive",
        "strided_slice_np_style",
        "sub",
        "sum",
        "tanh",
        "tensor_list_concat",
        "tensor_list_dynamic_shape",
        "tensor_list_get_item",
        "tensor_list_length",
        "tensor_list_resize",
        "tensor_list_set_item",
        "tensor_scatter_add",
        "tensor_scatter_update",
        "tile",
        "topk",
        "transpose",
        "transpose_conv",
        "unfused_gru",
        "unique",
        "unpack",
        "unroll_batch_matmul",
        "where",
        "where_v2",
        "while",
        "zeros_like",
    ]

def mlir_generated_test_denylisted_models():
    return [
        # TODO(b/150647400): This test passes in TF2 with tf.compat.v1 but
        # fails in TF1 with tf.compat.v1. Due to the testing environments
        # changing on 3/3, this will only be disabled temporarily.
        "unidirectional_sequence_lstm",
        "unidirectional_sequence_rnn",
    ]

# Test cases which only work internally now.
def no_oss_generated_test_models():
    return [
        "sparse_to_dense",
    ]

# List of models that fail generated tests for the conversion mode.
# If you have to disable a test, please add here with a link to the appropriate
# bug or issue.
def generated_test_models_failing(conversion_mode, delegate):
    if delegate == "xnnpack":
        # TODO(b/179802976): Revisit this list after XNNPack Delegate supports
        # dynamic tensors.
        return [
            "batch_to_space_nd",
            "broadcast_gradient_args",
            "broadcast_to",
            "concat",
            "cond",
            "conv2d_transpose",
            "conv3d_transpose",
            "depthwiseconv",
            "dynamic_rnn",
            "einsum",
            "expand_dims",
            "eye",
            "fill",
            "fully_connected",
            "fused_batch_norm",
            "gather",
            "gather_nd",
            "global_batch_norm",
            "leaky_relu",
            "mean",
            "mirror_pad",
            "multinomial",
            "one_hot",
            "pad",
            "padv2",
            "parse_example",
            "pow",
            "prelu",
            "random_standard_normal",
            "random_uniform",
            "range",
            "reduce_all",
            "reduce_any",
            "reduce_max",
            "reduce_min",
            "reduce_prod",
            "reshape",
            "roll",
            "roll_with_constant",
            "round",
            "scatter_nd",
            "segment_sum",
            "shape",
            "slice",
            "space_to_batch_nd",
            "squeeze",
            "static_hashtable",
            "stft",
            "strided_slice_1d_exhaustive",
            "strided_slice",
            "sum",
            "tensor_list_dynamic_shape",
            "tensor_list_length",
            "tensor_list_resize",
            "tile",
            "topk",
            "transpose",
            "unique",
            "where",
            "where_v2",
            "while",
        ]
    else:
        return []

def generated_test_models_successful(conversion_mode, delegate):
    """Returns the list of successful test models.

    Args:
      conversion_mode: Conversion mode.
      delegate: Delegate zip test runs with.

    Returns:
      List of successful test models for the conversion mode.
    """
    return [test_model for test_model in generated_test_models() if test_model not in generated_test_models_failing(conversion_mode, delegate)]

def merged_test_model_name():
    """Returns the name of merged test model.

    Returns:
      The name of merged test model.
    """
    return "merged_models"

def max_number_of_test_models_in_merged_zip():
    """Returns the maximum number of merged test models in a zip file.

    Returns:
      Maximum number of merged test models in a zip file.
    """
    return 15

def number_of_merged_zip_file(conversion_mode, delegate):
    """Returns the number of merged zip file targets.

    Returns:
      Number of merged zip file targets.
    """
    m = max_number_of_test_models_in_merged_zip()
    return (len(generated_test_models_successful(conversion_mode, delegate)) + m - 1) // m

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
        for delegate in generated_test_delegates():
            successful_tests = generated_test_models_successful(conversion_mode, delegate)
            if len(successful_tests) > 0:
                tags = common_test_tags_for_generated_models(conversion_mode, False)

                # Only non-merged tests are executed on TAP.
                # Merged test rules are only for running on the real device environment.
                if "notap" not in tags:
                    tags.append("notap")

                # Only execute merged tests on real device.
                if "no_oss" not in tags:
                    tags.append("no_oss")
                args = common_test_args_for_generated_models(conversion_mode, False)
                n = number_of_merged_zip_file(conversion_mode, delegate)
                for i in range(n):
                    test_i = "%s_%d" % (test, i)
                    options.append((conversion_mode, delegate, test_i, tags, args))
    return options

def flags_for_merged_test_models(test_name, conversion_mode, delegate):
    """Returns flags for generating zipped-example data file for merged tests.

    Args:
      test_name: str. Test name in the form of "<merged_model_name>_[<conversion_mode>_]%d".
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.
      delegate: str. Delegate zip test runs with.

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
    if delegate:
        index_string = index_string.replace("%s_" % delegate, "")

    # If the maximum number of test models in a file is 15 and the number of
    # successful test models are 62, 5 zip files will be generated.
    # To assign the test models fairly among these files, each zip file
    # should contain 12 or 13 test models. (62 / 5 = 12 ... 2)
    # Each zip file will have 12 test models and the first 2 zip files will have
    # 1 more test model each, resulting [13, 13, 12, 12, 12] assignment.
    # So Zip file 0, 1, 2, 3, 4 and 5 will have model[0:13], model[13:26],
    # model[26,38], model[38,50] and model[50,62], respectively.
    zip_index = int(index_string)
    num_merged_zips = number_of_merged_zip_file(conversion_mode, delegate)
    test_models = generated_test_models_successful(conversion_mode, delegate)

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
    for model in generated_test_models():
        if model not in denylisted_models:
            models.append(model)
    return models

def generated_test_conversion_modes():
    """Returns a list of conversion modes."""
    return ["with-flex", "forward-compat", "", "mlir-quant"]

def generated_test_delegates():
    """Returns a list of delegates."""
    return ["", "xnnpack"]

def delegate_suffix(delegate):
    """Returns the suffix for the delegate. Empty string for default (no delegate)."""
    if delegate:
        return "_%s" % delegate
    return ""

def generated_test_models_all():
    """Generates a list of all tests with the different converters.

    Returns:
      List of tuples representing:
            (conversion mode, delegate to use, name of test, test tags, test args).
    """
    conversion_modes = generated_test_conversion_modes()
    no_oss_tests = no_oss_generated_test_models()
    options = []
    for conversion_mode in conversion_modes:
        for delegate in generated_test_delegates():
            failing_tests = generated_test_models_failing(conversion_mode, delegate)
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
                options.append((conversion_mode, delegate, test, tags, args))

    return options

def common_test_args_for_generated_models(conversion_mode, failing):
    """Returns test args for generated model tests.

    Args:
      conversion_mode: Conversion mode.
      failing: True if the generated model test is failing.

    Returns:
      test args of generated models.
    """
    args = []

    # Flex conversion shouldn't suffer from the same conversion bugs
    # listed for the default TFLite kernel backend.
    if conversion_mode == "with-flex":
        args.append("--ignore_known_bugs=false")

    return args

def common_test_tags_for_generated_models(conversion_mode, failing):
    """Returns test tags for generated model tests.

    Args:
      conversion_mode: Conversion mode.
      failing: True if the generated model test is failing.

    Returns:
      tags for the failing generated model tests.
    """
    tags = []

    # Forward-compat coverage testing is largely redundant, and contributes
    # to coverage test bloat.
    if conversion_mode == "forward-compat":
        tags.append("nozapfhahn")

    if failing:
        return ["notap", "manual"]

    return tags

def gen_zip_test(
        name,
        test_name,
        conversion_mode,
        tags,
        args,
        delegate,
        **kwargs):
    """Generate a zipped-example test and its dependent zip files.

    Args:
      name: str. Resulting cc_test target name
      test_name: str. Test targets this model. Comes from the list above.
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.
      tags: tags for the generated cc_test.
      args: the basic cc_test args to be used.
      delegate: str. Delegate to use in the zip test.
      **kwargs: tf_cc_test kwargs
    """
    flags = ""

    if conversion_mode == "forward-compat":
        flags += " --make_forward_compat_test"
    elif conversion_mode == "mlir-quant":
        flags += " --mlir_quantizer"

    elif conversion_mode == "with-flex":
        flags += " --ignore_converter_errors --run_with_flex"
    if test_name.startswith(merged_test_model_name() + "_"):
        flags += flags_for_merged_test_models(test_name, conversion_mode, delegate)

    if delegate == "xnnpack":
        # buildifier: disable=list-append
        # Error: 'select' value has no field or method 'append'
        args += ["--use_xnnpack=true"]

        # TODO(b/204360746): XNNPack delegate don't support high dimension.
        flags += " --skip_high_dimension_inputs"

    zip_name = "zip_%s"
    zip_file = "%s.zip"
    if delegate:
        zip_name = "zip_%s_" + delegate
        zip_file = "%s_" + delegate + ".zip"
    gen_zipped_test_file(
        name = zip_name % test_name,
        file = zip_file % test_name,
        flags = flags,
    )
    tf_cc_test(name, tags = tags, args = args, **kwargs)

def gen_zipped_test_file(name, file, flags = ""):
    """Generate a zip file of tests by using :generate_examples.

    Args:
      name: str. Name of output. We will produce "`file`.files" as a target.
      file: str. The name of one of the generated_examples targets, e.g. "transpose"
      flags: str. Any additional flags to include
    """
    native.genrule(
        name = file + ".files",
        cmd = (("$(location //tensorflow/lite/testing:generate_examples) " +
                " --zip_to_output {0} {1} $(@D)").format(file, flags)),
        outs = [file],
        # `exec_tools` is required for PY3 compatibility in place of `tools`.
        exec_tools = [
            "//tensorflow/lite/testing:generate_examples",
        ],
    )

    native.filegroup(
        name = name,
        srcs = [file],
    )
