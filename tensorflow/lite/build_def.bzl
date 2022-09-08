"""Build macros for TF Lite."""

load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
    "if_oss",
    "tf_binary_additional_srcs",
    "tf_cc_shared_object",
)
load("//tensorflow/lite:special_rules.bzl", "tflite_copts_extra")
load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

def tflite_copts():
    """Defines common compile time flags for TFLite libraries."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
    ] + select({
        clean_dep("//tensorflow:android_arm"): [
            "-mfpu=neon",
        ],
        clean_dep("//tensorflow:ios_x86_64"): [
            "-msse4.1",
        ],
        clean_dep("//tensorflow:linux_x86_64"): [
            "-msse4.2",
        ],
        clean_dep("//tensorflow:linux_x86_64_no_sse"): [],
        clean_dep("//tensorflow:windows"): [
            "/DTFL_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
    }) + select({
        clean_dep("//tensorflow:optimized"): ["-O3"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow:android"): [
            "-ffunction-sections",  # Helps trim binary size.
            "-fdata-sections",  # Helps trim binary size.
        ],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-fno-exceptions",  # Exceptions are unused in TFLite.
        ],
    }) + select({
        clean_dep("//tensorflow/lite:tflite_with_xnnpack_explicit_false"): ["-DTFLITE_WITHOUT_XNNPACK"],
        "//conditions:default": [],
    }) + select({
        clean_dep("//tensorflow/lite:tensorflow_profiler_config"): ["-DTF_LITE_TENSORFLOW_PROFILER"],
        "//conditions:default": [],
    })

    return copts + tflite_copts_extra()

def tflite_copts_warnings():
    """Defines common warning flags used primarily by internal TFLite libraries."""

    # TODO(b/155906820): Include with `tflite_copts()` after validating clients.

    return select({
        clean_dep("//tensorflow:windows"): [
            # We run into trouble on Windows toolchains with warning flags,
            # as mentioned in the comments below on each flag.
            # We could be more aggressive in enabling supported warnings on each
            # Windows toolchain, but we compromise with keeping BUILD files simple
            # by limiting the number of config_setting's.
        ],
        "//conditions:default": [
            "-Wall",
        ],
    })

EXPORTED_SYMBOLS = clean_dep("//tensorflow/lite/java/src/main/native:exported_symbols.lds")
LINKER_SCRIPT = clean_dep("//tensorflow/lite/java/src/main/native:version_script.lds")

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
        clean_dep("//tensorflow:android"): [
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
            "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
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
        clean_dep("//tensorflow:android"): [
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def tflite_symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        clean_dep("//tensorflow:debug"): [],
        clean_dep("//tensorflow/lite:tflite_keep_symbols"): [],
        "//conditions:default": [
            # Omit symbol table, for all non debug builds
            "-Wl,-s",
        ],
    })

def tflite_linkopts_no_undefined():
    """Defines linker flags to enable errors for undefined symbols.

    This enables link-time errors for undefined symbols even when linking
    shared libraries, where the default behaviour on many systems is to only
    report errors for undefined symbols at runtime.
    """
    return if_oss(
        ["-Wl,--no-undefined"],
        select({
            # Can't enable errors for undefined symbols for asan/msan/tsan mode,
            # since undefined symbols in shared libraries (references to symbols
            # that will be defined in the main executable) are normal and
            # expected in those cases.
            "//tools/cpp:asan_build": [],
            "//tools/cpp:msan_build": [],
            "//tools/cpp:tsan_build": [],
            "//conditions:default": ["-Wl,--no-undefined"],
        }),
    )

def tflite_linkopts():
    """Defines linker flags for linking TFLite binary."""
    return tflite_linkopts_unstripped() + tflite_symbol_opts()

def tflite_jni_linkopts():
    """Defines linker flags for linking TFLite binary with JNI."""
    return tflite_jni_linkopts_unstripped() + tflite_symbol_opts()

def tflite_jni_binary(
        name,
        copts = tflite_copts(),
        linkopts = tflite_jni_linkopts(),
        linkscript = LINKER_SCRIPT,
        exported_symbols = EXPORTED_SYMBOLS,
        linkshared = 1,
        linkstatic = 1,
        testonly = 0,
        deps = [],
        tags = [],
        srcs = [],
        visibility = None):  # 'None' means use the default visibility.
    """Builds a jni binary for TFLite."""
    linkopts = linkopts + select({
        clean_dep("//tensorflow:macos"): [
            "-Wl,-exported_symbols_list,$(location {})".format(exported_symbols),
            "-Wl,-install_name,@rpath/" + name,
        ],
        clean_dep("//tensorflow:windows"): [],
        "//conditions:default": [
            "-Wl,--version-script,$(location {})".format(linkscript),
            "-Wl,-soname," + name,
        ],
    })
    native.cc_binary(
        name = name,
        copts = copts,
        linkshared = linkshared,
        linkstatic = linkstatic,
        deps = deps + [linkscript, exported_symbols],
        srcs = srcs,
        tags = tags,
        linkopts = linkopts,
        testonly = testonly,
        visibility = visibility,
    )

def tflite_cc_shared_object(
        name,
        copts = tflite_copts(),
        linkopts = [],
        linkstatic = 1,
        per_os_targets = False,
        **kwargs):
    """Builds a shared object for TFLite."""
    tf_cc_shared_object(
        name = name,
        copts = copts,
        linkstatic = linkstatic,
        linkopts = linkopts + tflite_jni_linkopts(),
        framework_so = [],
        per_os_targets = per_os_targets,
        **kwargs
    )

def tf_to_tflite(name, src, options, out):
    """Convert a frozen tensorflow graphdef to TF Lite's flatbuffer.

    Args:
      name: Name of rule.
      src: name of the input graphdef file.
      options: options passed to TFLite Converter.
      out: name of the output flatbuffer file.
    """

    toco_cmdline = " ".join([
        "$(location //tensorflow/lite/python:tflite_convert)",
        "--enable_v1_converter",
        ("--graph_def_file=$(location %s)" % src),
        ("--output_file=$(location %s)" % out),
    ] + options)
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = toco_cmdline,
        tools = ["//tensorflow/lite/python:tflite_convert"] + tf_binary_additional_srcs(),
    )

def DEPRECATED_tf_to_tflite(name, src, options, out):
    """DEPRECATED Convert a frozen tensorflow graphdef to TF Lite's flatbuffer, using toco.

    Please use tf_to_tflite instead.
    TODO(b/138396996): Migrate away from this deprecated rule.

    Args:
      name: Name of rule.
      src: name of the input graphdef file.
      options: options passed to TOCO.
      out: name of the output flatbuffer file.
    """

    toco_cmdline = " ".join([
        "$(location //tensorflow/lite/toco:toco)",
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
        tools = ["//tensorflow/lite/toco:toco"] + tf_binary_additional_srcs(),
    )

def tflite_to_json(name, src, out):
    """Convert a TF Lite flatbuffer to JSON.

    Args:
      name: Name of rule.
      src: name of the input flatbuffer file.
      out: name of the output JSON file.
    """

    flatc = "@flatbuffers//:flatc"
    schema = "//tensorflow/lite/schema:schema.fbs"
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
    schema = "//tensorflow/lite/schema:schema_fbs"
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

def _gen_selected_ops_impl(ctx):
    args = ctx.actions.args()
    args.add(ctx.attr.namespace, format = "--namespace=%s")
    args.add(ctx.outputs.output, format = "--output_registration=%s")
    tflite_path = "//tensorflow/lite"
    args.add("--tflite_path=%s" % tflite_path[2:])
    args.add_joined(
        ctx.files.models,
        join_with = ",",
        format_joined = "--input_models=%s",
    )

    ctx.actions.run(
        outputs = [ctx.outputs.output],
        inputs = ctx.files.models,
        arguments = [args],
        executable = ctx.executable._generate_op_registrations,
        mnemonic = "OpRegistration",
        progress_message = "gen_selected_ops",
    )

gen_selected_ops_rule = rule(
    implementation = _gen_selected_ops_impl,
    attrs = {
        "models": attr.label_list(default = [], allow_files = True),
        "namespace": attr.string(default = ""),
        "output": attr.output(),
        "_generate_op_registrations": attr.label(
            executable = True,
            default = Label(clean_dep(
                "//tensorflow/lite/tools:generate_op_registrations",
            )),
            cfg = "exec",
        ),
    },
)

def gen_selected_ops(name, model, namespace = "", **kwargs):
    """Generate the source file that includes only used ops.

    Args:
      name: Prefix of the generated source file.
      model: TFLite models to interpret, expect a list in case of multiple models.
      namespace: Namespace in which to put RegisterSelectedOps.
      **kwargs: Additional kwargs to pass to genrule.
    """

    # If there's only one model provided as a string.
    if type(model) == type(""):
        model = [model]

    gen_selected_ops_rule(
        name = name,
        models = model,
        namespace = namespace,
        output = name + "_registration.cc",
        **kwargs
    )

def flex_dep(target_op_sets):
    if "SELECT_TF_OPS" in target_op_sets:
        return ["//tensorflow/lite/delegates/flex:delegate"]
    else:
        return []

def gen_model_coverage_test(src, model_name, data, failure_type, tags, size = "medium"):
    """Generates Python test targets for testing TFLite models.

    Args:
      src: Main source file.
      model_name: Name of the model to test (must be also listed in the 'data'
        dependencies)
      data: List of BUILD targets linking the data.
      failure_type: List of failure types (none, toco, crash, inference, evaluation)
        expected for the corresponding combinations of op sets
        ("TFLITE_BUILTINS", "TFLITE_BUILTINS,SELECT_TF_OPS", "SELECT_TF_OPS").
      tags: List of strings of additional tags.
    """
    i = 0
    for target_op_sets in ["TFLITE_BUILTINS", "TFLITE_BUILTINS,SELECT_TF_OPS", "SELECT_TF_OPS"]:
        args = []
        if failure_type[i] != "none":
            args.append("--failure_type=%s" % failure_type[i])
        i = i + 1

        # Avoid coverage timeouts for large/enormous tests.
        coverage_tags = ["nozapfhahn"] if size in ["large", "enormous"] else []
        native.py_test(
            name = "model_coverage_test_%s_%s" % (model_name, target_op_sets.lower().replace(",", "_")),
            srcs = [src],
            main = src,
            size = size,
            args = [
                "--model_name=%s" % model_name,
                "--target_ops=%s" % target_op_sets,
            ] + args,
            data = data,
            srcs_version = "PY3",
            python_version = "PY3",
            tags = [
                "no_gpu",  # Executing with TF GPU configurations is redundant.
                "no_oss",
                "no_windows",
                # Disable sanitizer runs as models can be huge and can timeout.
                "noasan",
                "nomsan",
                "notsan",
            ] + tags + coverage_tags,
            deps = [
                "//third_party/py/tensorflow",
                "//tensorflow/lite/testing/model_coverage:model_coverage_lib",
                "//tensorflow/lite/python:lite",
                "//tensorflow/python:client_testlib",
            ] + flex_dep(target_op_sets),
        )

def tflite_custom_cc_library(
        name,
        models = [],
        srcs = [],
        deps = [],
        visibility = ["//visibility:private"],
        **kwargs):
    """Generates a tflite cc library, stripping off unused operators.

    This library includes the TfLite runtime as well as all operators needed for the given models.
    Op resolver can be retrieved using tflite::CreateOpResolver method.

    Args:
        name: Str, name of the target.
        models: List of models. This TFLite build will only include
            operators used in these models. If the list is empty, all builtin
            operators are included.
        srcs: List of files implementing custom operators if any.
        deps: Additional dependencies to build all the custom operators.
        visibility: Visibility setting for the generated target. Default to private.
        **kwargs: Additional arguments for native.cc_library.
    """
    real_srcs = []
    real_srcs.extend(srcs)
    real_deps = []
    real_deps.extend(deps)

    if models:
        gen_selected_ops(
            name = "%s_registration" % name,
            model = models,
            testonly = kwargs.get("testonly", default = False),
        )
        real_srcs.append(":%s_registration" % name)
        real_srcs.append("//tensorflow/lite:create_op_resolver_with_selected_ops.cc")
    else:
        # Support all operators if `models` not specified.
        real_deps.append("//tensorflow/lite:create_op_resolver_with_builtin_ops")

    native.cc_library(
        name = name,
        srcs = real_srcs,
        hdrs = [
            "//tensorflow/lite:create_op_resolver.h",
        ],
        copts = tflite_copts(),
        linkopts = select({
            "//tensorflow:windows": [],
            "//conditions:default": ["-lm", "-ldl"],
        }),
        deps = depset([
            "//tensorflow/lite:framework",
            "//tensorflow/lite/kernels:builtin_ops",
        ] + real_deps),
        visibility = visibility,
        **kwargs
    )

def tflite_custom_android_library(
        name,
        models = [],
        srcs = [],
        deps = [],
        custom_package = "org.tensorflow.lite",
        visibility = ["//visibility:private"],
        include_xnnpack_delegate = True,
        include_nnapi_delegate = True):
    """Generates a tflite Android library, stripping off unused operators.

    Note that due to a limitation in the JNI Java wrapper, the compiled TfLite shared binary
    has to be named as tensorflowlite_jni.so so please make sure that there is no naming conflict.
    i.e. you can't call this rule multiple times in the same build file.

    Args:
        name: Name of the target.
        models: List of models to be supported. This TFLite build will only include
            operators used in these models. If the list is empty, all builtin
            operators are included.
        srcs: List of files implementing custom operators if any.
        deps: Additional dependencies to build all the custom operators.
        custom_package: Name of the Java package. It is required by android_library in case
            the Java source file can't be inferred from the directory where this rule is used.
        visibility: Visibility setting for the generated target. Default to private.
        include_xnnpack_delegate: Whether to include the XNNPACK delegate or not.
        include_nnapi_delegate: Whether to include the NNAPI delegate or not.
    """
    tflite_custom_cc_library(name = "%s_cc" % name, models = models, srcs = srcs, deps = deps, visibility = visibility)

    delegate_deps = []
    if include_nnapi_delegate:
        delegate_deps.append("//tensorflow/lite/delegates/nnapi/java/src/main/native")
    if include_xnnpack_delegate:
        delegate_deps.append("//tensorflow/lite/delegates/xnnpack:xnnpack_delegate")

    # JNI wrapper expects a binary file called `libtensorflowlite_jni.so` in java path.
    tflite_jni_binary(
        name = "libtensorflowlite_jni.so",
        linkscript = "//tensorflow/lite/java:tflite_version_script.lds",
        # Do not sort: "native_framework_only" must come before custom tflite library.
        deps = [
            "//tensorflow/lite/java/src/main/native:native_framework_only",
            ":%s_cc" % name,
        ] + delegate_deps,
    )

    native.cc_library(
        name = "%s_jni" % name,
        srcs = ["libtensorflowlite_jni.so"],
        visibility = visibility,
    )

    android_library(
        name = name,
        manifest = "//tensorflow/lite/java:AndroidManifest.xml",
        srcs = ["//tensorflow/lite/java:java_srcs"],
        deps = [
            ":%s_jni" % name,
            "@org_checkerframework_qual",
        ],
        custom_package = custom_package,
        visibility = visibility,
    )

    aar_with_jni(
        name = "%s_aar" % name,
        android_library = name,
    )

def tflite_custom_c_library(
        name,
        models = [],
        **kwargs):
    """Generates a tflite cc library, stripping off unused operators.

    This library includes the C API and the op kernels used in the given models.

    Args:
        name: Str, name of the target.
        models: List of models. This TFLite build will only include
            operators used in these models. If the list is empty, all builtin
            operators are included.
       **kwargs: custom c_api cc_library kwargs.
    """
    op_resolver_deps = "//tensorflow/lite:create_op_resolver_with_builtin_ops"
    if models:
        gen_selected_ops(
            name = "%s_registration" % name,
            model = models,
            testonly = kwargs.get("testonly", default = False),
        )

        native.cc_library(
            name = "%s_create_op_resolver" % name,
            srcs = [
                ":%s_registration" % name,
            ],
            hdrs = ["//tensorflow/lite:create_op_resolver.h"],
            copts = tflite_copts(),
            deps = [
                "//tensorflow/lite:create_op_resolver_with_selected_ops",
                "//tensorflow/lite:op_resolver",
                "//tensorflow/lite:framework",
                "//tensorflow/lite/kernels:builtin_ops",
            ],
            # Using alwayslink here is needed, I believe, to avoid warnings about
            # backwards references when linking create_op_resolver_with_selected_ops,
            # which has a reference to the RegisterSelectedOps function defined by
            # '":%s_registration" % name' (the code generated by the call to
            # gen_selected_ops above).
            alwayslink = True,
            **kwargs
        )
        op_resolver_deps = "%s_create_op_resolver" % name

    native.cc_library(
        name = name,
        srcs = ["//tensorflow/lite/c:c_api_srcs"],
        hdrs = [
            "//tensorflow/lite/c:c_api.h",
            "//tensorflow/lite/c:c_api_experimental.h",
            "//tensorflow/lite/c:c_api_opaque.h",
        ],
        copts = tflite_copts(),
        deps = [
            op_resolver_deps,
            "//tensorflow/lite/c:common",
            "//tensorflow/lite/c:c_api_types",
            "//tensorflow/lite/kernels:kernel_util",
            "//tensorflow/lite:builtin_ops",
            "//tensorflow/lite:framework",
            "//tensorflow/lite:version",
            "//tensorflow/lite/core/api",
            "//tensorflow/lite/delegates:interpreter_utils",
            "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
            "//tensorflow/lite/kernels/internal:compatibility",
        ],
        **kwargs
    )

def tflite_combine_cc_tests(
        name,
        deps_conditions,
        extra_cc_test_tags = [],
        extra_build_test_tags = [],
        **kwargs):
    """Combine all certain cc_tests into a single cc_test and a build_test.

    Args:
      name: the name of the combined cc_test.
      deps_conditions: the list of deps that those cc_tests need to have in
          order to be combined.
      extra_cc_test_tags: the list of extra tags appended to the created
          combined cc_test.
      extra_build_test_tags: the list of extra tags appended to the created
          corresponding build_test for the combined cc_test.
      **kwargs: kwargs to pass to the cc_test rule of the test suite.
    """
    combined_test_srcs = {}
    combined_test_deps = {}
    for r in native.existing_rules().values():
        # We only include cc_test.
        if not r["kind"] == "cc_test":
            continue

        # Tests with data, args or special build option are not counted.
        if r["data"] or r["args"] or r["copts"] or r["defines"] or \
           r["includes"] or r["linkopts"] or r["additional_linker_inputs"]:
            continue

        # We only consider a single-src-file unit test.
        if len(r["srcs"]) > 1:
            continue

        dep_attr = r["deps"]
        if type(dep_attr) != type(()) and type(dep_attr) != type([]):
            # Attributes based on select() is not supported for simplicity.
            continue

        # The test has to depend on :test_main
        if not any([v in deps_conditions for v in dep_attr]):
            continue

        combined_test_srcs.update({s: True for s in r["srcs"]})
        combined_test_deps.update({d: True for d in r["deps"]})

    if combined_test_srcs:
        native.cc_test(
            name = name,
            size = "large",
            srcs = list(combined_test_srcs),
            tags = ["manual", "notap"] + extra_cc_test_tags,
            deps = list(combined_test_deps),
            **kwargs
        )
        build_test(
            name = "%s_build_test" % name,
            targets = [":%s" % name],
            tags = [
                "manual",
                "tflite_portable_build_test",
            ] + extra_build_test_tags,
        )

def tflite_self_contained_libs_test_suite(name):
    """Indicate that cc_library rules in this package *should* be self-contained.

    This adds build tests for each cc_library rule that verify that the
    library can be successfully linked with no undefined symbols.  It also
    adds a test_suite rule that contains all the generated build tests.

    Place this rule at the bottom of a package. Any cc_library rules that
    appear after the call to this rule will not be checked for undefined
    symbols.  Rules that are tagged with 'allow_undefined_symbols' in
    their 'tags' attribute will also not be checked for undefined symbols.

    Args:
      name: the name to use for the test_suite rule that contains
        the build tests generated by this macro.
    """
    build_tests = []

    for rule in native.existing_rules().values():
        rule_name = rule["name"]
        rule_kind = rule["kind"]
        rule_tags = rule["tags"]
        if rule_kind == "cc_library" and "allow_undefined_symbols" not in rule_tags:
            tflite_cc_shared_object(
                name = "%s_test_shared_lib" % rule_name,
                testonly = True,
                linkopts = tflite_linkopts_no_undefined(),
                deps = [":%s" % rule_name],
            )
            build_test(
                name = "%s_build_test" % rule_name,
                targets = ["%s_test_shared_lib" % rule_name],
            )
            build_tests.append("%s_build_test" % rule_name)

    native.test_suite(
        name = name,
        tests = build_tests,
    )
