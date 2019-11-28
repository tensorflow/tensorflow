# Description:
# Hadoop file system implementation.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
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
# 1. Ensure $JAVA_HOME is set to the location of a JDK 8 installation.
# 2. Download the binary Hadoop distribution from:
#    http://hadoop.apache.org/releases.html
# 3. Extract the Hadoop distribution and run:
#    source libexec/hadoop-config.sh
# 4. Optionally set up HDFS cluster configurations (optionally Kerberos) within
#    $HADOOP_HDFS_HOME/etc/hadoop if you want to test against real
#    distributed HDFS cluster
# 5. bazel test \
#      --test_env=LD_LIBRARY_PATH=$JAVA_HOME/jre/lib/amd64/server \
#      --test_env=HADOOP_HDFS_HOME=$HADOOP_HDFS_HOME \
#      --test_env=CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob) \
#      --test_strategy=local \
#      :hadoop_file_system_test
#    To test against the real distributed cluster, add the following option for
#    bazel test:
#      --test_env=HADOOP_TEST_TMPDIR=hdfs://cluster/test/tmp/dir
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
        "//tensorflow/core/platform:str_util",
    ],
)
