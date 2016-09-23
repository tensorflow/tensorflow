# Description:
# Hadoop file system implementation.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
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

cc_library(
    name = "hadoop_file_system",
    srcs = ["hadoop_file_system.cc"],
    hdrs = ["hadoop_file_system.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//third_party/hadoop:hdfs",
    ],
    alwayslink = 1,
)

# This test is set to manual because it requires downloading the Hadoop
# distribution to run. To run this test:
# 1. Ensure $JAVA_HOME is set.
# 2. Download the binary Hadoop distribution from:
#    http://hadoop.apache.org/releases.html
# 3. Extract the Hadoop distribution and run:
#    source libexec/hadoop-config.sh
# 4. bazel test \
#      --test_env=LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server \
#      --test_env=HADOOP_HDFS_HOME=$HADOOP_HDFS_HOME \
#      --test_env=CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob) \
#      --test_strategy=local \
#      :hadoop_file_system_test
tf_cc_test(
    name = "hadoop_file_system_test",
    size = "small",
    srcs = [
        "hadoop_file_system_test.cc",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":hadoop_file_system",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
