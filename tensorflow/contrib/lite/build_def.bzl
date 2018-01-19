"""Generate Flatbuffer binary from json."""

def tflite_copts():
  """Defines compile time flags."""
  copts = [
      "-DFARMHASH_NO_CXX_STRING",
  ] + select({
          "//tensorflow:android_arm64": [
              "-std=c++11",
              "-O3",
          ],
          "//tensorflow:android_arm": [
              "-mfpu=neon",
              "-mfloat-abi=softfp",
              "-std=c++11",
              "-O3",
          ],
          "//tensorflow:android_x86": [
              "-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK",
          ],
          "//tensorflow:ios_x86_64": [
              "-msse4.1",
          ],
          "//conditions:default": [],
  }) + select({
      "//tensorflow:with_default_optimizations": [],
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
      linkscript,
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

  toco = "//tensorflow/contrib/lite/toco:toco"
  native.genrule(
      name = name,
      srcs=[src, options],
      outs=[out],
      cmd = ("$(location %s) " +
             "   --input_file=$(location %s) " +
             "   --output_file=$(location %s) " +
             "   --input_format=TENSORFLOW_GRAPHDEF" +
             "   --output_format=TFLITE" +
             "   `cat $(location %s)`")
            % (toco, src, out, options),
      tools= [toco],
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

def gen_zipped_test_files(name, files):
  """Generate a zip file of tests by using :generate_examples.

  Args:
    name: Name of output. We will produce "`name`_files" as a target.
    files: A list of zip file basenames.
  """
  toco = "//tensorflow/contrib/lite/toco:toco"
  out_files = []
  for f in files:
    out_file = name + "/" + f
    out_files.append(out_file)
    native.genrule(
        name = name + "_" + f + ".files",
        cmd = ("$(locations :generate_examples) --toco $(locations %s) " % toco
               + " --zip_to_output " + f +
               " $(@D) zipped"),
        outs = [out_file],
        tools = [
            ":generate_examples",
            toco,
        ],
    )

  native.filegroup(
      name = name,
      srcs = out_files,
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
