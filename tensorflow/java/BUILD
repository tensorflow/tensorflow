# Description:
# TensorFlow Java API.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_java//java:defs.bzl", "java_library", "java_plugin")
load(
    "//tensorflow:tensorflow.bzl",
    "VERSION",
    "tf_binary_additional_srcs",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
    "tf_custom_op_library",
    "tf_java_test",
)
load(":build_defs.bzl", "JAVACOPTS")
load(":src/gen/gen_ops.bzl", "tf_java_op_gen_srcjar")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

java_library(
    name = "tensorflow",
    srcs = [
        ":java_op_sources",
        ":java_sources",
    ],
    data = tf_binary_additional_srcs() + [":libtensorflow_jni"],
    javacopts = JAVACOPTS,
    plugins = [":processor"],
    resources = [":java_resources"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "version-info",
    outs = ["src/main/resources/tensorflow-version-info"],
    cmd = "echo version=%s > $@" % VERSION,
    output_to_bindir = 1,
)

filegroup(
    name = "java_resources",
    srcs = [":version-info"],
    visibility = ["//visibility:private"],
)

# NOTE(ashankar): Rule to include the Java API in the Android Inference Library
# .aar. At some point, might make sense for a .aar rule here instead.
filegroup(
    name = "java_sources",
    srcs = glob([
        "src/main/java/org/tensorflow/*.java",
        "src/main/java/org/tensorflow/types/*.java",
    ]),
    visibility = ["//tensorflow/tools/android/inference_interface:__pkg__"],
)

java_plugin(
    name = "processor",
    generates_api = True,
    processor_class = "org.tensorflow.processor.OperatorProcessor",
    visibility = ["//visibility:public"],
    deps = [":processor_library"],
)

java_library(
    name = "processor_library",
    srcs = glob(["src/gen/java/org/tensorflow/processor/**/*.java"]),
    javacopts = JAVACOPTS,
    resources = glob(["src/gen/resources/META-INF/services/javax.annotation.processing.Processor"]),
    deps = [
        "@com_google_guava",
        "@com_squareup_javapoet",
    ],
)

filegroup(
    name = "java_op_sources",
    srcs = glob(["src/main/java/org/tensorflow/op/**/*.java"]) + [":java_op_gen_sources"],
    visibility = ["//visibility:private"],
)

tf_java_op_gen_srcjar(
    name = "java_op_gen_sources",
    api_def_srcs = [
        "//tensorflow/core/api_def:base_api_def",
        "//tensorflow/core/api_def:java_api_def",
    ],
    base_package = "org.tensorflow.op",
    gen_tool = ":java_op_gen_tool",
)

tf_cc_binary(
    name = "java_op_gen_tool",
    srcs = [
        "src/gen/cc/op_gen_main.cc",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    linkstatic = 1,
    deps = [
        ":java_op_gen_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "java_op_gen_lib",
    srcs = [
        "src/gen/cc/op_generator.cc",
        "src/gen/cc/op_specs.cc",
        "src/gen/cc/source_writer.cc",
    ],
    hdrs = [
        "src/gen/cc/java_defs.h",
        "src/gen/cc/op_generator.h",
        "src/gen/cc/op_specs.h",
        "src/gen/cc/source_writer.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = ["src/test/java/org/tensorflow/TestUtil.java"],
    javacopts = JAVACOPTS,
    deps = [":tensorflow"],
)

tf_java_test(
    name = "EagerSessionTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/EagerSessionTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.EagerSessionTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "EagerOperationBuilderTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/EagerOperationBuilderTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.EagerOperationBuilderTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "EagerOperationTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/EagerOperationTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.EagerOperationTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "GraphTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/GraphTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.GraphTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "GraphOperationBuilderTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/GraphOperationBuilderTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.GraphOperationBuilderTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "GraphOperationTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/GraphOperationTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.GraphOperationTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "SavedModelBundleTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/SavedModelBundleTest.java"],
    data = ["//tensorflow/cc/saved_model:saved_model_half_plus_two"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.SavedModelBundleTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "SessionTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/SessionTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.SessionTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "ShapeTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/ShapeTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.ShapeTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_custom_op_library(
    name = "my_test_op.so",
    srcs = ["src/test/native/my_test_op.cc"],
)

tf_java_test(
    name = "TensorFlowTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/TensorFlowTest.java"],
    data = [":my_test_op.so"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.TensorFlowTest",
    deps = [
        ":tensorflow",
        "@junit",
    ],
)

tf_java_test(
    name = "TensorTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/TensorTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.TensorTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "ScopeTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/ScopeTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.ScopeTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "PrimitiveOpTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/PrimitiveOpTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.PrimitiveOpTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "OperandsTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/OperandsTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.OperandsTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "ConstantTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/core/ConstantTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.core.ConstantTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "GeneratedOperationsTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/core/GeneratedOperationsTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.core.GeneratedOperationsTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "GradientsTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/core/GradientsTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.core.GradientsTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "ZerosTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/op/core/ZerosTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.op.core.ZerosTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

filegroup(
    name = "processor_test_resources",
    srcs = glob([
        "src/test/resources/org/tensorflow/**/*.java",
        "src/main/java/org/tensorflow/op/annotation/Operator.java",
    ]),
)

tf_cc_test(
    name = "source_writer_test",
    size = "small",
    srcs = [
        "src/gen/cc/source_writer_test.cc",
    ],
    data = [
        "src/gen/resources/test.java.snippet",
    ],
    deps = [
        ":java_op_gen_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

filegroup(
    name = "libtensorflow_jni",
    srcs = select({
        "//tensorflow:windows": [":tensorflow_jni.dll"],
        "//tensorflow:macos": [":libtensorflow_jni.dylib"],
        "//conditions:default": [":libtensorflow_jni.so"],
    }),
    visibility = ["//visibility:public"],
)

LINKER_VERSION_SCRIPT = ":config/version_script.lds"

LINKER_EXPORTED_SYMBOLS = ":config/exported_symbols.lds"

tf_cc_binary(
    name = "tensorflow_jni",
    # Set linker options to strip out anything except the JNI
    # symbols from the library. This reduces the size of the library
    # considerably (~50% as of January 2017).
    linkopts = select({
        "//tensorflow:debug": [],  # Disable all custom linker options in debug mode
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location {})".format(LINKER_EXPORTED_SYMBOLS),
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script,$(location {})".format(LINKER_VERSION_SCRIPT),
            # copybara:uncomment_begin(google-only)
            # "-Wl,--undefined-version",
            # copybara:uncomment_end
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    per_os_targets = True,
    deps = [
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/java/src/main/native",
        LINKER_VERSION_SCRIPT,
        LINKER_EXPORTED_SYMBOLS,
    ],
)

genrule(
    name = "pom",
    outs = ["pom.xml"],
    cmd = "$(location generate_pom) >$@",
    output_to_bindir = 1,
    tools = [":generate_pom"] + tf_binary_additional_srcs(),
)

tf_cc_binary(
    name = "generate_pom",
    srcs = ["generate_pom.cc"],
    deps = ["//tensorflow/c:c_api"],
)
