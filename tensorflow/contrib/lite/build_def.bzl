"""Generate Flatbuffer binary from json."""

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object",
    "tf_cc_test",
)

def tflite_copts():
    """Defines compile time flags."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
    ] + select({
        str(Label("//tensorflow:android_arm64")): [
            "-std=c++11",
            "-O3",
        ],
        str(Label("//tensorflow:android_arm")): [
            "-mfpu=neon",
            "-mfloat-abi=softfp",
            "-std=c++11",
            "-O3",
        ],
        str(Label("//tensorflow:android_x86")): [
            "-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK",
        ],
        str(Label("//tensorflow:ios_x86_64")): [
            "-msse4.1",
        ],
        str(Label("//tensorflow:windows")): [
            "/DTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [],
    }) + select({
        str(Label("//tensorflow:with_default_optimizations")): [],
        "//conditions:default": ["-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK"],
    })

    return copts

LINKER_SCRIPT = "//tensorflow/contrib/lite/java/src/main/native:version_script.lds"

def tflite_linkopts_unstripped():
    """Defines linker flags to reduce size of TFLite binary.

       These are useful when trying to investigate the relative size of the
       symbols in TFLite.

    Returns:
       a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        "//tensorflow:android": [
            "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export.
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def tflite_jni_linkopts_unstripped():
    """Defines linker flags to reduce size of TFLite binary with JNI.

       These are useful when trying to investigate the relative size of the
       symbols in TFLite.

    Returns:
       a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        "//tensorflow:android": [
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def tflite_linkopts():
    """Defines linker flags to reduce size of TFLite binary."""
    return tflite_linkopts_unstripped() + select({
        "//tensorflow:android": [
            "-s",  # Omit symbol table.
        ],
        "//conditions:default": [],
    })

def tflite_jni_linkopts():
    """Defines linker flags to reduce size of TFLite binary with JNI."""
    return tflite_jni_linkopts_unstripped() + select({
        "//tensorflow:android": [
            "-s",  # Omit symbol table.
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
        ],
        "//conditions:default": [],
    })

def tflite_jni_binary(
        name,
        copts = tflite_copts(),
        linkopts = tflite_jni_linkopts(),
        linkscript = LINKER_SCRIPT,
        linkshared = 1,
        linkstatic = 1,
        deps = []):
    """Builds a jni binary for TFLite."""
    linkopts = linkopts + [
        "-Wl,--version-script",  # Export only jni functions & classes.
        "$(location {})".format(linkscript),
    ]
    native.cc_binary(
        name = name,
        copts = copts,
        linkshared = linkshared,
        linkstatic = linkstatic,
        deps = deps + [linkscript],
        linkopts = linkopts,
    )

def tflite_cc_shared_object(
        name,
        copts = tflite_copts(),
        linkopts = [],
        linkstatic = 1,
        deps = []):
    """Builds a shared object for TFLite."""
    tf_cc_shared_object(
        name = name,
        copts = copts,
        linkstatic = linkstatic,
        linkopts = linkopts + tflite_jni_linkopts(),
        framework_so = [],
        deps = deps,
    )

def tf_to_tflite(name, src, options, out):
    """Convert a frozen tensorflow graphdef to TF Lite's flatbuffer.

    Args:
      name: Name of rule.
      src: name of the input graphdef file.
      options: options passed to TOCO.
      out: name of the output flatbuffer file.
    """

    toco_cmdline = " ".join([
        "//tensorflow/contrib/lite/toco:toco",
        "--input_format=TENSORFLOW_GRAPHDEF",
        "--output_format=TFLITE",
        ("--input_file=$(location %s)" % src),
        ("--output_file=$(location %s)" % out),
    ] + options)
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = toco_cmdline,
        tools = ["//tensorflow/contrib/lite/toco:toco"],
    )

def tflite_to_json(name, src, out):
    """Convert a TF Lite flatbuffer to JSON.

    Args:
      name: Name of rule.
      src: name of the input flatbuffer file.
      out: name of the output JSON file.
    """

    flatc = "@flatbuffers//:flatc"
    schema = "//tensorflow/contrib/lite/schema:schema.fbs"
    native.genrule(
        name = name,
        srcs = [schema, src],
        outs = [out],
        cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.bin &&" +
               "$(location %s) --raw-binary --strict-json -t" +
               " -o /tmp $(location %s) -- $${TMP}.bin &&" +
               "cp $${TMP}.json $(location %s)") %
              (src, flatc, schema, out),
        tools = [flatc],
    )

def json_to_tflite(name, src, out):
    """Convert a JSON file to TF Lite's flatbuffer.

    Args:
      name: Name of rule.
      src: name of the input JSON file.
      out: name of the output flatbuffer file.
    """

    flatc = "@flatbuffers//:flatc"
    schema = "//tensorflow/contrib/lite/schema:schema_fbs"
    native.genrule(
        name = name,
        srcs = [schema, src],
        outs = [out],
        cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.json &&" +
               "$(location %s) --raw-binary --unknown-json --allow-non-utf8 -b" +
               " -o /tmp $(location %s) $${TMP}.json &&" +
               "cp $${TMP}.bin $(location %s)") %
              (src, flatc, schema, out),
        tools = [flatc],
    )

# This is the master list of generated examples that will be made into tests. A
# function called make_XXX_tests() must also appear in generate_examples.py.
# Disable a test by commenting it out. If you do, add a link to a bug or issue.
def generated_test_models():
    return [
        "add",
        "arg_min_max",
        "avg_pool",
        "batch_to_space_nd",
        "concat",
        "constant",
        "control_dep",
        "conv",
        "conv_with_shared_weights",
        "conv_to_depthwiseconv_with_shared_weights",
        "depthwiseconv",
        "div",
        "equal",
        "exp",
        "expand_dims",
        "floor",
        "floor_div",
        "fully_connected",
        "fused_batch_norm",
        "gather",
        "global_batch_norm",
        "greater",
        "greater_equal",
        "sum",
        "l2norm",
        "l2_pool",
        "less",
        "less_equal",
        "local_response_norm",
        "log_softmax",
        "log",
        "logical_and",
        "logical_or",
        "logical_xor",
        "lstm",
        "max_pool",
        "maximum",
        "mean",
        "minimum",
        "mul",
        "neg",
        "not_equal",
        "one_hot",
        "pack",
        "pad",
        "padv2",
        "prelu",
        "pow",
        "reduce_any",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "relu",
        "relu1",
        "relu6",
        "reshape",
        "resize_bilinear",
        "rsqrt",
        "shape",
        "sigmoid",
        "sin",
        "slice",
        "softmax",
        "space_to_batch_nd",
        "space_to_depth",
        "sparse_to_dense",
        "split",
        "sqrt",
        "square",
        "squeeze",
        "strided_slice",
        "strided_slice_1d_exhaustive",
        "sub",
        "tile",
        "topk",
        "transpose",
        #"transpose_conv",   # disabled due to b/111213074
        "unpack",
        "where",
        "zeros_like",
    ]

def generated_test_conversion_modes():
    """Returns a list of conversion modes."""

    # TODO(nupurgarg): Add "pb2lite" when it's in open source. b/113614050.
    return ["toco-flex", ""]

def generated_test_models_all():
    """Generates a list of all tests with the different converters.

    Returns:
      List of tuples representing (conversion mode, name of test).
    """
    conversion_modes = generated_test_conversion_modes()
    tests = generated_test_models()
    options = []
    for conversion_mode in conversion_modes:
        for test in tests:
            if conversion_mode:
                test += "_%s" % conversion_mode
            options.append((conversion_mode, test))
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
    toco = "//tensorflow/contrib/lite/toco:toco"
    flags = ""
    if conversion_mode:
        # TODO(nupurgarg): Comment in when pb2lite is in open source. b/113614050.
        # if conversion_mode == "pb2lite":
        #     toco = "//tensorflow/contrib/lite/experimental/pb2lite:pb2lite"
        flags = "--ignore_toco_errors --run_with_flex"
        kwargs["tags"].append("skip_already_failing")
        kwargs["tags"].append("no_oss")
        kwargs["tags"].append("notap")

    gen_zipped_test_file(
        name = "zip_%s" % test_name,
        file = "%s.zip" % test_name,
        toco = toco,
        flags = flags,
    )
    tf_cc_test(name, **kwargs)

def gen_zipped_test_file(name, file, toco, flags):
    """Generate a zip file of tests by using :generate_examples.

    Args:
      name: str. Name of output. We will produce "`file`.files" as a target.
      file: str. The name of one of the generated_examples targets, e.g. "transpose"
      toco: str. Pathname of toco binary to run
      flags: str. Any additional flags to include
    """
    native.genrule(
        name = file + ".files",
        cmd = (("$(locations :generate_examples) --toco $(locations {0}) " +
                " --zip_to_output {1} {2} $(@D)").format(toco, file, flags)),
        outs = [file],
        tools = [
            ":generate_examples",
            toco,
        ],
    )

    native.filegroup(
        name = name,
        srcs = [file],
    )

def gen_selected_ops(name, model):
    """Generate the library that includes only used ops.

    Args:
      name: Name of the generated library.
      model: TFLite model to interpret.
    """
    out = name + "_registration.cc"
    tool = "//tensorflow/contrib/lite/tools:generate_op_registrations"
    tflite_path = "//tensorflow/contrib/lite"
    native.genrule(
        name = name,
        srcs = [model],
        outs = [out],
        cmd = ("$(location %s) --input_model=$(location %s) --output_registration=$(location %s) --tflite_path=%s") %
              (tool, model, out, tflite_path[2:]),
        tools = [tool],
    )

def gen_full_model_test(conversion_modes, models, data, test_suite_tag):
    """Generates Python test targets for testing TFLite models.

    Args:
      conversion_modes: List of conversion modes to test the models on.
      models: List of models to test.
      data: List of BUILD targets linking the data.
      test_suite_tag: Tag identifying the model test suite.
    """
    options = [
        (conversion_mode, model)
        for model in models
        for conversion_mode in conversion_modes
    ]

    for conversion_mode, model_name in options:
        native.py_test(
            name = "model_coverage_test_%s_%s" % (model_name, conversion_mode.lower()),
            srcs = ["model_coverage_test.py"],
            main = "model_coverage_test.py",
            args = [
                "--model_name=%s" % model_name,
                "--converter_mode=%s" % conversion_mode,
            ],
            data = data,
            srcs_version = "PY2AND3",
            tags = [
                "no_oss",
                "no_windows",
                "notap",
            ] + [test_suite_tag],
            deps = [
                "//tensorflow/contrib/lite/testing:model_coverage_lib",
                "//tensorflow/contrib/lite/python:lite",
                "//tensorflow/python:client_testlib",
            ],
        )
