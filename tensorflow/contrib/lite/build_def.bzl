"""Generate Flatbuffer binary from json."""
load(
    "//tensorflow:tensorflow.bzl",
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
  return select({
      "//tensorflow:android": [
          "-Wl,--no-export-dynamic", # Only inc syms referenced by dynamic obj.
          "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export.
          "-Wl,--gc-sections", # Eliminate unused code and data.
          "-Wl,--as-needed", # Don't link unused libs.
      ],
      "//tensorflow/contrib/lite:mips": [],
      "//tensorflow/contrib/lite:mips64": [],
      "//conditions:default": [
          "-Wl,--icf=all",  # Identical code folding.
      ],
  })

def tflite_jni_linkopts_unstripped():
  """Defines linker flags to reduce size of TFLite binary with JNI.

     These are useful when trying to investigate the relative size of the
     symbols in TFLite.

  Returns:
     a select object with proper linkopts
  """
  return select({
      "//tensorflow:android": [
          "-Wl,--gc-sections", # Eliminate unused code and data.
          "-Wl,--as-needed", # Don't link unused libs.
      ],
      "//tensorflow/contrib/lite:mips": [],
      "//tensorflow/contrib/lite:mips64": [],
      "//conditions:default": [
          "-Wl,--icf=all",  # Identical code folding.
      ],
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

def tflite_jni_binary(name,
                      copts=tflite_copts(),
                      linkopts=tflite_jni_linkopts(),
                      linkscript=LINKER_SCRIPT,
                      linkshared=1,
                      linkstatic=1,
                      deps=[]):
  """Builds a jni binary for TFLite."""
  linkopts = linkopts + [
      "-Wl,--version-script",  # Export only jni functions & classes.
      "$(location {})".format(linkscript),
  ]
  native.cc_binary(
      name=name,
      copts=copts,
      linkshared=linkshared,
      linkstatic=linkstatic,
      deps= deps + [linkscript],
      linkopts=linkopts)

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
  ] + options )
  native.genrule(
      name = name,
      srcs=[src],
      outs=[out],
      cmd = toco_cmdline,
      tools= ["//tensorflow/contrib/lite/toco:toco"],
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
      cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.bin &&"  +
             "$(location %s) --raw-binary --strict-json -t" +
             " -o /tmp $(location %s) -- $${TMP}.bin &&" +
             "cp $${TMP}.json $(location %s)")
            % (src, flatc, schema, out),
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
      cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.json &&"  +
             "$(location %s) --raw-binary --unknown-json --allow-non-utf8 -b" +
             " -o /tmp $(location %s) $${TMP}.json &&" +
             "cp $${TMP}.bin $(location %s)")
      % (src, flatc, schema, out),
      tools = [flatc],
  )

# This is the master list of generated examples that will be made into tests. A
# function called make_XXX_tests() must also appear in generate_examples.py.
# Disable a test by commenting it out. If you do, add a link to a bug or issue.
def generated_test_models():
    return [
        "add",
        "arg_max",
        "avg_pool",
        "batch_to_space_nd",
        "concat",
        "constant",
        "control_dep",
        # "conv",
        "depthwiseconv",
        "div",
        "equal",
        "exp",
        "expand_dims",
        "floor",
        "fully_connected",
        "fused_batch_norm",
        "gather",
        "global_batch_norm",
        "greater",
        "greater_equal",
        "l2norm",
        "l2_pool",
        "less",
        "less_equal",
        "local_response_norm",
        "log_softmax",
        "lstm",
        "max_pool",
        "maximum",
        "mean",
        "minimum",
        "mul",
        "neg",
        "not_equal",
        "pad",
        "padv2",
        # "prelu",
        "relu",
        "relu1",
        "relu6",
        "reshape",
        "resize_bilinear",
        "sigmoid",
        "sin",
        "slice",
        "softmax",
        "space_to_batch_nd",
        "space_to_depth",
        "sparse_to_dense",
        "split",
        "squeeze",
        "strided_slice",
        "strided_slice_1d_exhaustive",
        "sub",
        "tile",
        "topk",
        "transpose",
        "transpose_conv",
        "where",
    ]

def gen_zip_test(name, test_name, **kwargs):
  """Generate a zipped-example test and its dependent zip files.

  Args:
    name: Resulting cc_test target name
    test_name: Test targets this model. Comes from the list above.
    **kwargs: tf_cc_test kwargs.
  """
  gen_zipped_test_file(
      name = "zip_%s" % test_name,
      file = "%s.zip" % test_name,
  )
  tf_cc_test(name, **kwargs)

def gen_zipped_test_file(name, file):
  """Generate a zip file of tests by using :generate_examples.

  Args:
    name: Name of output. We will produce "`file`.files" as a target.
    file: The name of one of the generated_examples targets, e.g. "transpose"
  """
  toco = "//tensorflow/contrib/lite/toco:toco"
  native.genrule(
      name = file + ".files",
      cmd = ("$(locations :generate_examples) --toco $(locations %s) " % toco
             + " --zip_to_output " + file + " $(@D)"),
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
      cmd = ("$(location %s) --input_model=$(location %s) --output_registration=$(location %s) --tflite_path=%s")
      % (tool, model, out, tflite_path[2:]),
      tools = [tool],
  )
