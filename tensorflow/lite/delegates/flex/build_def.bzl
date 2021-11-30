"""Generate custom flex delegate library."""

load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
    "if_android",
    "if_ios",
    "if_mobile",
    "tf_cc_binary",
    "tf_copts",
    "tf_defines_nortti_if_lite_protos",
    "tf_features_nolayering_check_if_android_or_ios",
    "tf_features_nomodules_if_mobile",
    "tf_opts_nortti_if_lite_protos",
    "tf_portable_full_lite_protos",
)
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_cc_shared_object",
    "tflite_copts",
    "tflite_jni_binary",
    "tflite_jni_linkopts",
)
load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/lite:special_rules.bzl", "flex_portable_tensorflow_deps")

def generate_flex_kernel_header(
        name,
        models,
        testonly = 0,
        additional_deps = []):
    """A rule to generate a header file listing only used operators.

    Args:
      name: Name of the generated library.
      models: TFLite models to interpret.
      testonly: Should be marked as true if additional_deps is testonly.
      additional_deps: Dependencies for additional TF ops.

    Returns:
      A struct with 'header' and 'include_path' fields that
      contain the generated header and the required include entry.
    """
    include_path = "%s_tf_generated_kernel_header" % name
    header = include_path + "/ops_to_register.h"

    if type(models) != type([]):
        models = [models]

    # List all flex ops from models.
    model_file_args = " --graphs=%s" % ",".join(
        ["$(location %s)" % f for f in models],
    )
    list_ops_output = include_path + "/list_flex_ops"
    list_ops_tool = clean_dep("//tensorflow/lite/tools:list_flex_ops_main")
    if additional_deps:
        tf_cc_binary(
            name = "%s_list_flex_ops_main" % name,
            deps = [
                clean_dep("//tensorflow/lite/tools:list_flex_ops_main_lib"),
            ] + additional_deps,
            testonly = testonly,
        )
        list_ops_tool = ":%s_list_flex_ops_main" % name
    native.genrule(
        name = "%s_list_flex_ops" % name,
        srcs = models,
        outs = [list_ops_output],
        tools = [list_ops_tool],
        message = "Listing flex ops from %s..." % ",".join(models),
        cmd = ("$(location " + list_ops_tool + ")" +
               model_file_args + " > \"$@\""),
        testonly = testonly,
    )

    # Generate the kernel registration header file from list of flex ops.
    tool = clean_dep("//tensorflow/python/tools:print_selective_registration_header")
    native.genrule(
        name = "%s_kernel_registration" % name,
        srcs = [list_ops_output],
        outs = [header],
        tools = [tool],
        message = "Processing %s..." % list_ops_output,
        cmd = ("$(location " + tool + ")" +
               " --default_ops=\"\"" +
               " --proto_fileformat=ops_list" +
               " --graphs=" + "$(location " + list_ops_output + ") > \"$@\""),
    )
    return struct(include_path = include_path, header = header)

def tflite_flex_cc_library(
        name,
        models = [],
        additional_deps = [],
        testonly = 0,
        visibility = ["//visibility:public"],
        link_symbol = True):
    """A rule to generate a flex delegate with only ops to run listed models.

    Args:
      name: Name of the generated flex delegate.
      models: TFLite models to interpret. The library will only include ops and kernels
          to support these models. If empty, the library will include all Tensorflow
          ops and kernels.
      additional_deps: Dependencies for additional TF ops.
      testonly: Mark this library as testonly if true.
      visibility: visibility of the generated rules.
    """
    portable_tensorflow_lib = clean_dep("//tensorflow/core:portable_tensorflow_lib")
    if models:
        CUSTOM_KERNEL_HEADER = generate_flex_kernel_header(
            name = "%s_tf_op_headers" % name,
            models = models,
            additional_deps = additional_deps,
            testonly = testonly,
        )

        # Define a custom tensorflow_lib with selective registration.
        # The library will only contain ops exist in provided models.
        native.cc_library(
            name = "%s_tensorflow_lib" % name,
            srcs = if_mobile([
                clean_dep("//tensorflow/core:portable_op_registrations_and_gradients"),
                clean_dep("//tensorflow/core/kernels:portable_core_ops"),
                clean_dep("//tensorflow/core/kernels:portable_extended_ops"),
            ]) + [CUSTOM_KERNEL_HEADER.header],
            copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
            defines = [
                "SELECTIVE_REGISTRATION",
                "SUPPORT_SELECTIVE_REGISTRATION",
            ] + tf_portable_full_lite_protos(
                full = [],
                lite = ["TENSORFLOW_LITE_PROTOS"],
            ) + tf_defines_nortti_if_lite_protos(),
            features = tf_features_nomodules_if_mobile() + tf_features_nolayering_check_if_android_or_ios(),
            linkopts = if_android(["-lz"]) + if_ios(["-lz"]),
            includes = [
                CUSTOM_KERNEL_HEADER.include_path,
            ],
            textual_hdrs = [
                clean_dep("//tensorflow/core/kernels:portable_all_ops_textual_hdrs"),
            ],
            visibility = visibility,
            deps = flex_portable_tensorflow_deps() + [
                clean_dep("//tensorflow/core:protos_all_cc"),
                clean_dep("//tensorflow/core:portable_tensorflow_lib_lite"),
                clean_dep("//tensorflow/core/platform:strong_hash"),
                clean_dep("//tensorflow/lite/delegates/flex:portable_images_lib"),
            ],
            alwayslink = 1,
            testonly = testonly,
        )
        portable_tensorflow_lib = ":%s_tensorflow_lib" % name

    delegate_symbol = []
    if link_symbol:
        delegate_symbol.append(clean_dep("//tensorflow/lite/delegates/flex:delegate_symbol"))

    # Define a custom flex delegate with above tensorflow_lib.
    native.cc_library(
        name = name,
        hdrs = [
            clean_dep("//tensorflow/lite/delegates/flex:delegate.h"),
        ],
        features = tf_features_nolayering_check_if_android_or_ios(),
        visibility = visibility,
        deps = [
            clean_dep("//tensorflow/lite/delegates/flex:delegate_data"),
            clean_dep("//tensorflow/lite/delegates/flex:delegate_only_runtime"),
            clean_dep("//tensorflow/lite/delegates/utils:simple_delegate"),
        ] + select({
            clean_dep("//tensorflow:android"): [
                portable_tensorflow_lib,
            ],
            clean_dep("//tensorflow:ios"): [
                portable_tensorflow_lib,
            ],
            "//conditions:default": [
                clean_dep("//tensorflow/core:tensorflow"),
                clean_dep("//tensorflow/lite/c:common"),
            ],
        }) + additional_deps + delegate_symbol,
        testonly = testonly,
        alwayslink = 1,
    )

def tflite_flex_shared_library(
        name,
        models = [],
        additional_deps = [],
        testonly = 0,
        visibility = ["//visibility:private"]):
    """A rule to generate a flex delegate shared library with only ops to run listed models.

    The output library name is platform dependent:
    - Linux/Android: `lib{name}.so`
    - Mac: `lib{name}.dylib`
    - Windows: `lib{name}.dll`

    Args:
      name: Name of the library.
      models: TFLite models to interpret. The library will only include ops and kernels
          to support these models. If empty, the library will include all Tensorflow
          ops and kernels.
      additional_deps: Dependencies for additional TF ops.
      testonly: Mark this library as testonly if true.
      visibility: visibility of the generated rules.
    """
    tflite_flex_cc_library(
        name = "%s_flex_delegate" % name,
        models = models,
        additional_deps = additional_deps,
        testonly = testonly,
        visibility = visibility,
    )

    tflite_cc_shared_object(
        name = name,
        # Until we have more granular symbol export for the C++ API on Windows,
        # export all symbols.
        features = ["windows_export_all_symbols"],
        linkopts = select({
            "//tensorflow:macos": [
                "-Wl,-exported_symbols_list,$(location //tensorflow/lite/delegates/flex:exported_symbols.lds)",
            ],
            "//tensorflow:windows": [],
            "//conditions:default": [
                "-Wl,-z,defs",
                "-Wl,--version-script,$(location //tensorflow/lite/delegates/flex:version_script.lds)",
            ],
        }),
        per_os_targets = True,
        deps = [
            "%s_flex_delegate" % name,
            "//tensorflow/lite/delegates/flex:exported_symbols.lds",
            "//tensorflow/lite/delegates/flex:version_script.lds",
        ],
    )

def tflite_flex_jni_library(
        name,
        models = [],
        additional_deps = [],
        testonly = 0,
        visibility = ["//visibility:private"]):
    """A rule to generate a jni library listing only used operators.

    The libtensorflowlite_flex_jni.so name is fixed due to a limitation in JNI
    Java wrapper, so please make sure there is no naming conflicts.

    Args:
      name: Prefix of the generated libraries.
      models: TFLite models to interpret. The library will only include ops and kernels
          to support these models. If empty, the library will include all Tensorflow
          ops and kernels.
      additional_deps: Dependencies for additional TF ops.
      testonly: Mark this library as testonly if true.
      visibility: visibility of the generated rules.
    """

    # Define a custom flex_delegate that depends on above tensorflow_lib.
    # This will reduce the binary size comparing to the original flex delegate.
    tflite_flex_cc_library(
        name = "%s_flex_delegate" % name,
        models = models,
        additional_deps = additional_deps,
        testonly = testonly,
        visibility = visibility,
    )

    # Define a custom flex_native that depends on above flex_delegate.
    native.cc_library(
        name = "%s_flex_native" % name,
        srcs = [
            clean_dep("//tensorflow/lite/testing:init_tensorflow.h"),
            clean_dep("//tensorflow/lite/testing:init_tensorflow.cc"),
            clean_dep("//tensorflow/lite/delegates/flex/java/src/main/native:flex_delegate_jni.cc"),
        ],
        copts = tflite_copts(),
        testonly = testonly,
        visibility = visibility,
        deps = [
            ":%s_flex_delegate" % name,
            clean_dep("//tensorflow/lite/java/jni"),
            clean_dep("//tensorflow/lite/delegates/utils:simple_delegate"),
        ] + select({
            clean_dep("//tensorflow:android"): [],
            clean_dep("//tensorflow:ios"): [],
            "//conditions:default": [
                clean_dep("//tensorflow/core:lib"),
            ],
        }),
        alwayslink = 1,
    )

    # Build the jni binary based on the above flex_native.
    # The library name is fixed as libtensorflowlite_flex_jni.so in FlexDelegate.java.
    tflite_jni_binary(
        name = "libtensorflowlite_flex_jni.so",
        linkopts = tflite_jni_linkopts(),
        testonly = testonly,
        deps = [
            ":%s_flex_native" % name,
        ],
    )

def tflite_flex_android_library(
        name,
        models = [],
        additional_deps = [],
        custom_package = "org.tensorflow.lite.flex",
        testonly = 0,
        visibility = ["//visibility:private"]):
    """A rule to generate an android library based on the selective-built jni library.

    Args:
      name: name of android library.
      models: TFLite models used for selective build. The library will only include ops
          and kernels to support these models. If empty, the library will include all
          Tensorflow ops and kernels.
      additional_deps: Dependencies for additional TF ops.
      custom_package: Java package for which java sources will be generated.
      testonly: Mark this library as testonly if true.
      visibility: visibility of the generated rules.
    """
    tflite_flex_jni_library(
        name = name,
        models = models,
        additional_deps = additional_deps,
        testonly = testonly,
        visibility = visibility,
    )

    native.cc_library(
        name = "%s_native" % name,
        srcs = ["libtensorflowlite_flex_jni.so"],
        testonly = testonly,
        visibility = visibility,
    )

    android_library(
        name = name,
        srcs = [clean_dep("//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate")],
        manifest = clean_dep("//tensorflow/lite/java:AndroidManifest.xml"),
        proguard_specs = [clean_dep("//tensorflow/lite/java:proguard.flags")],
        custom_package = custom_package,
        testonly = testonly,
        deps = [
            ":%s_native" % name,
            clean_dep("//tensorflow/lite/java:tensorflowlite_java"),
            clean_dep("@org_checkerframework_qual"),
        ],
        visibility = visibility,
    )
