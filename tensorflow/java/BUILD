# Description:
# TensorFlow Java API.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

load(":build_defs.bzl", "JAVACOPTS")
load(":src/gen/gen_ops.bzl", "tf_java_op_gen_srcjar")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_binary_additional_srcs",
    "tf_cc_binary",
    "tf_copts",
    "tf_custom_op_library",
    "tf_java_test",
    "tf_cc_test",
)

java_library(
    name = "tensorflow",
    srcs = [
        ":java_op_sources",
        ":java_sources",
    ],
    data = [":libtensorflow_jni"],
    javacopts = JAVACOPTS,
    plugins = [":processor"],
    visibility = ["//visibility:public"],
)

# NOTE(ashankar): Rule to include the Java API in the Android Inference Library
# .aar. At some point, might make sense for a .aar rule here instead.
filegroup(
    name = "java_sources",
    srcs = glob([
        "src/main/java/org/tensorflow/*.java",
        "src/main/java/org/tensorflow/types/*.java",
    ]),
    visibility = [
        "//tensorflow/contrib/android:__pkg__",
        "//tensorflow/java:__pkg__",
    ],
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
)

filegroup(
    name = "java_op_sources",
    srcs = glob(["src/main/java/org/tensorflow/op/**/*.java"]) + [
        ":java_op_gen_sources",
    ],
    visibility = [
        "//tensorflow/java:__pkg__",
    ],
)

tf_java_op_gen_srcjar(
    name = "java_op_gen_sources",
    gen_base_package = "org.tensorflow.op",
    gen_tool = "java_op_gen_tool",
    ops_libs = [
        "array_ops",
        "candidate_sampling_ops",
        "control_flow_ops",
        "data_flow_ops",
        "image_ops",
        "io_ops",
        "linalg_ops",
        "logging_ops",
        "math_ops",
        "nn_ops",
        "no_op",
        "parsing_ops",
        "random_ops",
        "sparse_ops",
        "state_ops",
        "string_ops",
        "training_ops",
        "user_ops",
    ],
)

# Build the gen tool as a library, as it will be linked to a core/ops binary
# file before making it an executable. See tf_java_op_gen_srcjar().
cc_library(
    name = "java_op_gen_tool",
    srcs = [
        "src/gen/cc/op_gen_main.cc",
    ],
    copts = tf_copts(),
    deps = [
        ":java_op_gen_lib",
    ],
)

cc_library(
    name = "java_op_gen_lib",
    srcs = [
        "src/gen/cc/op_generator.cc",
        "src/gen/cc/source_writer.cc",
    ],
    hdrs = [
        "src/gen/cc/java_defs.h",
        "src/gen/cc/op_generator.h",
        "src/gen/cc/source_writer.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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
    name = "OperationBuilderTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/OperationBuilderTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.OperationBuilderTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "@junit",
    ],
)

tf_java_test(
    name = "OperationTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/OperationTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.OperationTest",
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
        "src/gen/resources/test.snippet.java",
    ],
    deps = [
        ":java_op_gen_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

filegroup(
    name = "libtensorflow_jni",
    srcs = select({
        "//tensorflow:darwin": [":libtensorflow_jni.dylib"],
        "//conditions:default": [":libtensorflow_jni.so"],
    }),
    visibility = ["//visibility:public"],
)

LINKER_VERSION_SCRIPT = ":config/version_script.lds"

LINKER_EXPORTED_SYMBOLS = ":config/exported_symbols.lds"

tf_cc_binary(
    name = "libtensorflow_jni.so",
    # Set linker options to strip out anything except the JNI
    # symbols from the library. This reduces the size of the library
    # considerably (~50% as of January 2017).
    linkopts = select({
        "//tensorflow:debug": [],  # Disable all custom linker options in debug mode
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by LINKER_EXPORTED_SYMBOLS
            LINKER_EXPORTED_SYMBOLS,
        ],
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by LINKER_VERSION_SCRIPT
            LINKER_VERSION_SCRIPT,
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
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

# System.loadLibrary() on OS X looks for ".dylib" or ".jnilib"
# and no ".so". If and when https://github.com/bazelbuild/bazel/issues/914
# is resolved, perhaps this workaround rule can be removed.
genrule(
    name = "darwin-compat",
    srcs = [":libtensorflow_jni.so"],
    outs = ["libtensorflow_jni.dylib"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
)
