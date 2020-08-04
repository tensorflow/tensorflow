"""Generate custom flex delegate library."""

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_ios",
    "if_mobile",
    "tf_cc_binary",
    "tf_copts",
    "tf_defines_nortti_if_lite_protos",
    "tf_features_nomodules_if_mobile",
    "tf_opts_nortti_if_lite_protos",
    "tf_portable_full_lite_protos",
)
load(
    "//tensorflow/lite:build_def.bzl",
    "tflite_copts",
    "tflite_jni_binary",
    "tflite_jni_linkopts",
)
load("@build_bazel_rules_android//android:rules.bzl", "android_library")

def generate_flex_kernel_header(
        name,
        models,
        additional_deps = []):
    """A rule to generate a header file listing only used operators.

    Args:
      name: Name of the generated library.
      models: TFLite models to interpret.
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
    list_ops_tool = "//tensorflow/lite/tools:list_flex_ops_main"
    if additional_deps:
        tf_cc_binary(
            name = "%s_list_flex_ops_main" % name,
            deps = [
                "//tensorflow/lite/tools:list_flex_ops_main_lib",
            ] + additional_deps,
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
    )

    # Generate the kernel registration header file from list of flex ops.
    tool = "//tensorflow/python/tools:print_selective_registration_header"
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
        visibility = ["//visibility:public"]):
    """A rule to generate a flex delegate with only ops to run listed models.

    Args:
      name: Name of the generated flex delegate.
      models: TFLite models to interpret. The library will only include ops and kernels
          to support these models. If empty, the library will include all Tensorflow
          ops and kernels.
      additional_deps: Dependencies for additional TF ops.
      visibility: visibility of the generated rules.
    """
    portable_tensorflow_lib = "//tensorflow/core:portable_tensorflow_lib"
    if models:
        CUSTOM_KERNEL_HEADER = generate_flex_kernel_header(
            name = "%s_tf_op_headers" % name,
            models = models,
            additional_deps = additional_deps,
        )

        # Define a custom tensorflow_lib with selective registration.
        # The library will only contain ops exist in provided models.
        native.cc_library(
            name = "%s_tensorflow_lib" % name,
            srcs = if_mobile([
                "//tensorflow/core:portable_op_registrations_and_gradients",
                "//tensorflow/core/kernels:android_core_ops",
                "//tensorflow/core/kernels:android_extended_ops",
            ]) + [CUSTOM_KERNEL_HEADER.header],
            copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_lite_protos() + if_ios(["-Os"]),
            defines = [
                "SELECTIVE_REGISTRATION",
                "SUPPORT_SELECTIVE_REGISTRATION",
            ] + tf_portable_full_lite_protos(
                full = [],
                lite = ["TENSORFLOW_LITE_PROTOS"],
            ) + tf_defines_nortti_if_lite_protos(),
            features = tf_features_nomodules_if_mobile(),
            linkopts = if_android(["-lz"]) + if_ios(["-lz"]),
            includes = [
                CUSTOM_KERNEL_HEADER.include_path,
            ],
            textual_hdrs = [
                "//tensorflow/core/kernels:android_all_ops_textual_hdrs",
            ],
            visibility = visibility,
            deps = [
                "@com_google_absl//absl/strings:str_format",
                "//third_party/fft2d:fft2d_headers",
                "//third_party/eigen3",
                "@com_google_absl//absl/types:optional",
                "@gemmlowp",
                "//tensorflow/core:protos_all_cc",
                "//tensorflow/core:portable_tensorflow_lib_lite",
                "//tensorflow/core/platform:strong_hash",
            ],
            alwayslink = 1,
        )
        portable_tensorflow_lib = ":%s_tensorflow_lib" % name

    # Define a custom flex delegate with above tensorflow_lib.
    native.cc_library(
        name = name,
        hdrs = [
            "//tensorflow/lite/delegates/flex:delegate.h",
        ],
        visibility = visibility,
        deps = [
            "//tensorflow/lite/delegates/flex:delegate_data",
            "//tensorflow/lite/delegates/flex:delegate_only_runtime",
            "//tensorflow/lite/delegates/utils:simple_delegate",
        ] + select({
            "//tensorflow:android": [
                portable_tensorflow_lib,
            ],
            "//tensorflow:ios": [
                portable_tensorflow_lib,
            ],
            "//conditions:default": [
                "//tensorflow/core:tensorflow",
                "//tensorflow/lite/c:common",
            ],
        }) + additional_deps,
        alwayslink = 1,
    )

def tflite_flex_jni_library(
        name,
        models = [],
        additional_deps = [],
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
      visibility: visibility of the generated rules.
    """

    # Define a custom flex_delegate that depends on above tensorflow_lib.
    # This will reduce the binary size comparing to the original flex delegate.
    tflite_flex_cc_library(
        name = "%s_flex_delegate" % name,
        models = models,
        additional_deps = additional_deps,
        visibility = visibility,
    )

    # Define a custom flex_native that depends on above flex_delegate.
    native.cc_library(
        name = "%s_flex_native" % name,
        srcs = [
            "//tensorflow/lite/testing:init_tensorflow.h",
            "//tensorflow/lite/testing:init_tensorflow.cc",
            "//tensorflow/lite/delegates/flex/java/src/main/native:flex_delegate_jni.cc",
        ],
        copts = tflite_copts(),
        visibility = visibility,
        deps = [
            ":%s_flex_delegate" % name,
            "//tensorflow/lite/java/jni",
            "//tensorflow/lite/delegates/utils:simple_delegate",
        ] + select({
            "//tensorflow:android": [],
            "//tensorflow:ios": [],
            "//conditions:default": [
                "//tensorflow/core:lib",
            ],
        }),
        alwayslink = 1,
    )

    # Build the jni binary based on the above flex_native.
    # The library name is fixed as libtensorflowlite_flex_jni.so in FlexDelegate.java.
    tflite_jni_binary(
        name = "libtensorflowlite_flex_jni.so",
        linkopts = tflite_jni_linkopts(),
        deps = [
            ":%s_flex_native" % name,
        ],
    )

def tflite_flex_android_library(
        name,
        models = [],
        custom_package = "org.tensorflow.lite.flex",
        visibility = ["//visibility:private"]):
    """A rule to generate an android library based on the selective-built jni library.

    Args:
      name: name of android library.
      models: TFLite models used for selective build. The library will only include ops
          and kernels to support these models. If empty, the library will include all
          Tensorflow ops and kernels.
      custom_package: Java package for which java sources will be generated.
      visibility: visibility of the generated rules.
    """
    tflite_flex_jni_library(
        name = name,
        models = models,
        visibility = visibility,
    )

    native.cc_library(
        name = "%s_native" % name,
        srcs = ["libtensorflowlite_flex_jni.so"],
        visibility = visibility,
    )

    android_library(
        name = name,
        srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
        manifest = "//tensorflow/lite/java:AndroidManifest.xml",
        proguard_specs = ["//tensorflow/lite/java:proguard.flags"],
        custom_package = custom_package,
        deps = [
            ":%s_native" % name,
            "//tensorflow/lite/java:tensorflowlite_java",
            "@org_checkerframework_qual",
        ],
        visibility = visibility,
    )
