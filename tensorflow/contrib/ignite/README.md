# Apache Ignite Integration

-   [Overview](#overview)
-   [Features](#features)
    *   [Distributed In-Memory Datasource](#distributed-in-memory-datasource)
    *   [Structured Objects](#structured-objects)
    *   [Distributed Training](#distributed-training)
    *   [Distributed File System](#distributed-file-system)
    *   [SSL Connection](#ssl-connection)
    *   [Windows Support](#windows-support)
-   [Try it out](#try-it-out)
    *   [Ignite Dataset](#ignite-dataset)
    *   [IGFS](#igfs)
-   [Limitations](#limitations)

## Overview

[Apache Ignite](https://ignite.apache.org/) is a memory-centric distributed
database, caching, and processing platform for transactional, analytical, and
streaming workloads, delivering in-memory speeds at petabyte scale. This contrib
package contains an integration between Apache Ignite and TensorFlow. The
integration is based on
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data) from TensorFlow
side and
[Binary Client Protocol](https://apacheignite.readme.io/v2.6/docs/binary-client-protocol)
from Apache Ignite side. It allows to use Apache Ignite as a data source for
neural network training, inference and all other computations supported by
TensorFlow. Another part of this module is an integration with distributed file
system based on Apache Ignite.

## Features

Ignite Dataset provides features that you can use in a wide range of cases. The
most important and interesting features are described below.

### Distributed In-Memory Datasource
[Apache Ignite](https://ignite.apache.org/) is a distributed in-memory database, caching, and processing platform that provides fast data access. It allows you to avoid limitations of hard drive and store and operate with as much data as you need in distributed cluster. You can utilize
these benefits of Apache Ignite by using Ignite Dataset. Moreover, Ignite Dataset can be used for the following use-cases:
- If you have a **gigabyte** of data you can keep it on a single machine on a hard drive, but you will face with hard drive speed limitations. At the same time, you can store your data in Apache Ignite on the same machine and use it as a datasource for TensorFlow and thus avoid these limitations.
- If you have a **terabyte** of data you probably still can keep it on a single machine on a hard drive, but you will face with hard drive speed limitations again. At the same time, you can store your data in Apache Ignite distributed in-memory cluster and use it as a datasource for TensorFlow and thus avoid these limitations.
- If you have a **petabyte** of data you can't keep it on a single machine. At the same time, you can store your data in Apache Ignite distributed in-memory cluster and use it as a datasource for TensorFlow.

Note that Apache Ignite is not just a step of ETL pipeline between a database or a data warehouse and TensorFlow. Apache Ignite is a high-grade database itself. By choosing Apache Ignite and TensorFlow you are getting everything you need to work with operational or historical data and, at the same time, an ability to use this data for neural network training and inference.

```bash
$ apache-ignite-fabric/bin/ignite.sh
$ apache-ignite-fabric/bin/sqlline.sh -u "jdbc:ignite:thin://localhost:10800/"

jdbc:ignite:thin://localhost/> CREATE TABLE KITTEN_CACHE (ID LONG PRIMARY KEY, NAME VARCHAR);
jdbc:ignite:thin://localhost/> INSERT INTO KITTEN_CACHE VALUES (1, 'WARM KITTY');
jdbc:ignite:thin://localhost/> INSERT INTO KITTEN_CACHE VALUES (2, 'SOFT KITTY');
jdbc:ignite:thin://localhost/> INSERT INTO KITTEN_CACHE VALUES (3, 'LITTLE BALL OF FUR');
```

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> tf.enable_eager_execution()
>>>
>>> dataset = IgniteDataset(cache_name="SQL_PUBLIC_KITTEN_CACHE")
>>>
>>> for element in dataset:
>>>   print(element)

{'key': 1, 'val': {'NAME': b'WARM KITTY'}}
{'key': 2, 'val': {'NAME': b'SOFT KITTY'}}
{'key': 3, 'val': {'NAME': b'LITTLE BALL OF FUR'}}
```

### Structured Objects
[Apache Ignite](https://ignite.apache.org/) allows to store any type of objects. These objects can have any hierarchy. Ignite Dataset provides an ability to work with such objects.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> tf.enable_eager_execution()
>>>
>>> dataset = IgniteDataset(cache_name="IMAGES")
>>>
>>> for element in dataset.take(1):
>>>   print(element)

{
    'key': 'kitten.png',
    'val': {
        'metadata': {
            'file_name': b'kitten.png',
            'label': b'little ball of fur',
            width: 800,
            height: 600
        },
        'pixels': [0, 0, 0, 0, ..., 0]
    }
}
```
 Neural network training and other computations require transformations that can be done as part of [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) pipeline if you use Ignite Dataset.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> tf.enable_eager_execution()
>>>
>>> dataset = IgniteDataset(cache_name="IMAGES").map(lambda obj: obj['val']['pixels'])
>>>
>>> for element in dataset:
>>>   print(element)

[0, 0, 0, 0, ..., 0]
```

### Distributed Training

TensorFlow is a machine learning framework that [natively supports](https://www.tensorflow.org/deploy/distributed) distributed neural network training, inference and other computations. The main idea behind the distributed neural network training is the ability to calculate gradients of loss functions (squares of the errors) on every partition of data (in terms of horizontal partitioning) and then sum them to get loss function gradient of the whole dataset. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla[\sum_1^n(y&space;-&space;\hat{y})^2]&space;=&space;\nabla[\sum_1^{n_1}(y&space;-&space;\hat{y})^2]&space;&plus;&space;\nabla[\sum_{n_1}^{n_2}(y&space;-&space;\hat{y})^2]&space;&plus;&space;...&space;&plus;&space;\nabla[\sum_{n_{k-1}}^n(y&space;-&space;\hat{y})^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla[\sum_1^n(y&space;-&space;\hat{y})^2]&space;=&space;\nabla[\sum_1^{n_1}(y&space;-&space;\hat{y})^2]&space;&plus;&space;\nabla[\sum_{n_1}^{n_2}(y&space;-&space;\hat{y})^2]&space;&plus;&space;...&space;&plus;&space;\nabla[\sum_{n_{k-1}}^n(y&space;-&space;\hat{y})^2]" title="\nabla[\sum_1^n(y - \hat{y})^2] = \nabla[\sum_1^{n_1}(y - \hat{y})^2] + \nabla[\sum_{n_1}^{n_2}(y - \hat{y})^2] + ... + \nabla[\sum_{n_{k-1}}^n(y - \hat{y})^2]" /></a>

Using this ability we can calculate gradients on the nodes the data is stored on, reduce them and then finally update model parameters. It allows to avoid data transfers between nodes and thus to avoid network bottlenecks.

Apache Ignite uses horizontal partitioning to store data in distributed cluster. When we create Apache Ignite cache (or table in terms of SQL), we can specify the number of partitions the data will be partitioned on. For example, if an Apache Ignite cluster consists of 10 machines and we create cache with 10 partitions, then every machine will maintain approximately one data partition.

Ignite Dataset allows using these two aspects of distributed neural network
training (using TensorFlow) and Apache Ignite partitioning. Ignite Dataset is a
computation graph operation that can be performed on a remote worker. The remote
worker can override Ignite Dataset parameters (such as `host`, `port` or `part`)
by setting correspondent environment variables for worker process (such as
`IGNITE_DATASET_HOST`, `IGNITE_DATASET_PORT` or `IGNITE_DATASET_PART`). Using
this overriding approach, we can assign a specific partition to every worker so
that one worker handles one partition and, at the same time, transparently work
with single dataset.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>>
>>> dataset = IgniteDataset("IMAGES")
>>>
>>> # Compute gradients locally on every worker node.
>>> gradients = []
>>> for i in range(5):
>>>     with tf.device("/job:WORKER/task:%d" % i):
>>>         device_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
>>>         device_next_obj = device_iterator.get_next()
>>>         gradient = compute_gradient(device_next_obj)
>>>         gradients.append(gradient)
>>>
>>> # Aggregate them on master node.
>>> result_gradient = tf.reduce_sum(gradients)
>>>
>>> with tf.Session("grpc://localhost:10000") as sess:
>>>     print(sess.run(result_gradient))
```

High-level TensorFlow API for [distributed training](https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy) is supported as well.

### Distributed File System

In addition to database functionality Apache Ignite provides a distributed file
system called [IGFS](https://ignite.apache.org/features/igfs.html). IGFS
delivers a similar functionality to Hadoop HDFS, but only in-memory. In fact, in
addition to its own APIs, IGFS implements Hadoop FileSystem API and can be
transparently plugged into Hadoop or Spark deployments. This contrib package
contains an integration between IGFS and TensorFlow. The integration is based on
[custom filesystem plugin](https://www.tensorflow.org/extend/add_filesys) from
TensorFlow side and
[IGFS Native API](https://ignite.apache.org/features/igfs.html) from Apache
Ignite side. It has numerous uses, for example:

*   Checkpoints of state can be saved to IGFS for reliability and
    fault-tolerance.
*   Training processes communicate with TensorBoard by writing event files to a
    directory, which TensorBoard watches. IGFS allows this communication to work
    even when TensorBoard runs in a different process or machine.

### SSL Connection

Apache Ignite allows to protect data transfer channels by
[SSL](https://en.wikipedia.org/wiki/Transport_Layer_Security) and
authentication. Ignite Dataset supports both SSL connection with and without
authentication. For more information, please refer to the
[Apache Ignite SSL/TLS](https://apacheignite.readme.io/docs/ssltls)
documentation.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> tf.enable_eager_execution()
>>>
>>> dataset = IgniteDataset(cache_name="IMAGES",
                            certfile="client.pem",
                            cert_password="password",
                            username="ignite",
                            password="ignite")
```

### Windows Support

Ignite Dataset is fully compatible with Windows. You can use it as part of TensorFlow on your Windows workstation as well as on Linux/MacOS systems.

## Try it out

Following examples will help you to easily start working with this module.

### Ignite Dataset

The simplest way to try Ignite Dataset is to run a
[Docker](https://www.docker.com/) container with Apache Ignite and loaded
[MNIST](http://yann.lecun.com/exdb/mnist/) data and after start interrupt with
it using Ignite Dataset. Such container is available on Docker Hub:
[dmitrievanthony/ignite-with-mnist](https://hub.docker.com/r/dmitrievanthony/ignite-with-mnist/).
You need to start this container on your machine:

```
docker run -it -p 10800:10800 dmitrievanthony/ignite-with-mnist
```

After that you will be able to work with it following way:

![ignite-dataset-mnist](https://s3.amazonaws.com/helloworld23423423ew23/ignite-dataset-mnist-2.png "Ignite Dataset Mnist")

### IGFS

The simplest way to try IGFS with TensorFlow is to run
[Docker](https://www.docker.com/) container with Apache Ignite and enabled IGFS
and then interrupt with it using TensorFlow
[tf.gfile](https://www.tensorflow.org/api_docs/python/tf/gfile). Such container
is available on Docker Hub:
[dmitrievanthony/ignite-with-igfs](https://hub.docker.com/r/dmitrievanthony/ignite-with-igfs/).
You need to start this container on your machine:

```
docker run -it -p 10500:10500 dmitrievanthony/ignite-with-igfs
```

After that you will be able to work with it following way:

```python
>>> import tensorflow as tf
>>> import tensorflow.contrib.ignite.python.ops.igfs_ops
>>>
>>> with tf.gfile.Open("igfs:///hello.txt", mode='w') as w:
>>>   w.write("Hello, world!")
>>>
>>> with tf.gfile.Open("igfs:///hello.txt", mode='r') as r:
>>>   print(r.read())

Hello, world!
```

## Limitations

Presently, Ignite Dataset works with assumption that all objects in the cache have the same structure (homogeneous objects) and the cache contains at least one object. Another limitation concerns structured objects, Ignite Dataset does not support UUID, Maps and Object arrays that might be parts of an object structure.
