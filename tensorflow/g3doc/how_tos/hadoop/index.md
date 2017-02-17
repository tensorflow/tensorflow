# How to run TensorFlow on Hadoop

This document describes how to run TensorFlow on Hadoop. It will be expanded to
describe running on various cluster managers, but only describes running on HDFS
at the moment.

## HDFS

We assume that you are familiar with [reading data](../reading_data/index.md).

To use HDFS with TensorFlow, change the file paths you use to read and write
data to an HDFS path. For example:

```python
filename_queue = tf.train.string_input_producer([
    "hdfs://namenode:8020/path/to/file1.csv",
    "hdfs://namenode:8020/path/to/file2.csv",
])
```

If you want to use the namenode specified in your HDFS configuration files, then
change the file prefix to `hdfs://default/`.

When launching your TensorFlow program, the following environment variables must
be set:

*   **JAVA_HOME**: The location of your Java installation.
*   **HADOOP_HDFS_HOME**: The location of your HDFS installation. You can also
    set this environment variable by running:

    ```shell
    source ${HADOOP_HOME}/libexec/hadoop-config.sh
    ```

*   **LD_LIBRARY_PATH**: To include the path to libjvm.so, and optionally the path 
    to libhdfs.so if your Hadoop distribution does not install libhdfs.so in 
    `$HADOOP_HDFS_HOME/lib/native`. On Linux:

    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
    ```

*   **CLASSPATH**: The Hadoop jars must be added prior to running your
    TensorFlow program. The CLASSPATH set by
    `${HADOOP_HOME}/libexec/hadoop-config.sh` is insufficient. Globs must be
    expanded as described in the libhdfs documentation:

    ```shell
    CLASSPATH=$($HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python your_script.py
    ```
    For older version of Hadoop/libhdfs (older than 2.6.0), you have to expand the
    classpath wildcard manually. For more details, see
    [HADOOP-10903](https://issues.apache.org/jira/browse/HADOOP-10903).

If the Hadoop cluster is in secure mode, the following environment variable must
be set:

*   **KERB_TICKET_CACHE_PATH**: The path of Kerberos ticket cache file. For example:

    ```shell
    export KERB_TICKET_CACHE_PATH=/tmp/krb5cc_10002
    ```

If you are running [Distributed TensorFlow](../distributed/index.md), then all
workers must have the environment variables set and Hadoop installed.
