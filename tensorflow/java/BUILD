# Description:
# TensorFlow Java API.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

java_library(
    name = "tensorflow",
    srcs = glob(["src/main/java/org/tensorflow/*.java"]),
    data = [":libtensorflow-jni"],
    visibility = ["//visibility:public"],
)

java_library(
    name = "testutil",
    testonly = 1,
    srcs = ["src/test/java/org/tensorflow/TestUtil.java"],
    deps = [":tensorflow"],
)

java_test(
    name = "GraphTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/GraphTest.java"],
    test_class = "org.tensorflow.GraphTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "//external:junit",
    ],
)

java_test(
    name = "OperationBuilderTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/OperationBuilderTest.java"],
    test_class = "org.tensorflow.OperationBuilderTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "//external:junit",
    ],
)

java_test(
    name = "SessionTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/SessionTest.java"],
    test_class = "org.tensorflow.SessionTest",
    deps = [
        ":tensorflow",
        ":testutil",
        "//external:junit",
    ],
)

java_test(
    name = "TensorFlowTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/TensorFlowTest.java"],
    test_class = "org.tensorflow.TensorFlowTest",
    deps = [
        ":tensorflow",
        "//external:junit",
    ],
)

java_test(
    name = "TensorTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/TensorTest.java"],
    test_class = "org.tensorflow.TensorTest",
    deps = [
        ":tensorflow",
        "//external:junit",
    ],
)

filegroup(
    name = "libtensorflow-jni",
    srcs = select({
        "//tensorflow:darwin": [":libtensorflow-jni.dylib"],
        "//conditions:default": [":libtensorflow-jni.so"],
    }),
)

cc_binary(
    name = "libtensorflow-jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = ["//tensorflow/java/src/main/native"],
)

# System.loadLibrary() on OS X looks for ".dylib" or ".jnilib"
# and no ".so". If and when https://github.com/bazelbuild/bazel/issues/914
# is resolved, perhaps this workaround rule can be removed.
genrule(
    name = "darwin-compat",
    srcs = [":libtensorflow-jni.so"],
    outs = ["libtensorflow-jni.dylib"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
