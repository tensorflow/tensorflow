# IGFS (Ignite File System)

- [Overview](#overview)
- [Try it out](#try-it-out)

## Overview

[Apache Ignite](https://ignite.apache.org/) is a memory-centric distributed database, caching, and processing platform for
transactional, analytical, and streaming workloads, delivering in-memory speeds at petabyte scale. In addition to database functionality Apache Ignite provides a distributed file system called [IGFS](https://ignite.apache.org/features/igfs.html). IGFS delivers a similar functionality to Hadoop HDFS, but only in-memory. In fact, in addition to its own APIs, IGFS implements Hadoop FileSystem API and can be transparently plugged into Hadoop or Spark deployments. This contrib package contains an intergration between IGFS and TensorFlow. The integration is based on [custom filesystem plugin](https://www.tensorflow.org/extend/add_filesys) from TensorFlow side and [IGFS Native API](https://ignite.apache.org/features/igfs.html) from Apache Ignite side. It has numerous uses, for example:
* Checkpoints of state can be saved to IGFS for reliability and fault-tolerance.
* Training processes communicate with TensorBoard by writing event files to a directory, which TensorBoard watches. IGFS allows this communication to work even when TensorBoard runs in a different process or machine.

## Try it out

The simplest way to try IGFS with TensorFlow is to run [Docker](https://www.docker.com/) container with Apache Ignite and enabled IGFS and then interruct with it using TensorFlow [tf.gfile](https://www.tensorflow.org/api_docs/python/tf/gfile). Such container is available on Docker Hub: [dmitrievanthony/ignite-with-igfs](https://hub.docker.com/r/dmitrievanthony/ignite-with-igfs/). You need to start this container on your machine:

```
docker run -it -p 10500:10500 dmitrievanthony/ignite-with-igfs
```

After that you will be able to work with it following way:

```python
>>> import tensorflow as tf
>>> import tensorflow.contrib.igfs.python.ops.igfs_ops
>>> 
>>> with tf.gfile.Open("igfs:///hello.txt", mode='w') as w:
>>>   w.write("Hello, world!")
>>>
>>> with tf.gfile.Open("igfs:///hello.txt", mode='r') as r:
>>>   print(r.read())

Hello, world!
```
