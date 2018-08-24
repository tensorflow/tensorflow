### Ignite Dataset
# Ignite Dataset

- [Overview](#overview)
- [Features](#features)
  * [Distributed In-Memory Datasource](#distributed-in-memory-datasource)
  * [Structured Objects](#structured-objects)
  * [Distributed Training](#distributed-training)
  * [SSL Connection](#ssl-connection)
  * [Windows Support](#windows-support)
- [Try it out](#try-it-out)
- [Limitations](#limitations)

## Overview

[Apache Ignite](https://ignite.apache.org/) is a memory-centric distributed database, caching, and processing platform for
transactional, analytical, and streaming workloads, delivering in-memory speeds at petabyte scale. This contrib package contains an integration between Apache Ignite and TensorFlow. The integration is based on [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) from TensorFlow side and [Binary Client Protocol](https://apacheignite.readme.io/v2.6/docs/binary-client-protocol) from Apache Ignite side. It allows to use Apache Ignite as a datasource for neural network training, inference and all other computations supported by TensorFlow. 

## Features

Ignite Dataset provides a set of features that makes it possible to use it in a wide range of cases. The most important and interesting features are described below.

### Distributed In-Memory Datasource
[Apache Ignite](https://ignite.apache.org/) is a distributed in-memory database, caching, and processing platform that allows to avoid limitations of hard drive and provide high reading speed and ability to store and operate with as much data as you need in distributed cluster. Using of Ignite Dataset makes it possible to utilize all these advantages. 
- If you have a **gigabyte** of data you can keep it on a single machine on a hard drive, but you will face with hard drive speed limitations. At the same time, you can store your data in Apache Ignite on the same machine and use it as a datasource for TensorFlow and thus avoid these limitations.
- If you have a **terabyte** of data you probably still can keep it on a single machine on a hard drive, but you will face with hard drive speed limitations again. At the same time, you can store your data in Apache Ignite distributed in-memory cluster and use it as a datasource for TensorFlow and thus avoid these limitations.
- If you have a **petabyte** of data you can't keep it on a single machine. At the same time, you can store your data in Apache Ignite distributed in-memory cluster and use it as a datasource for TensorFlow.

It's  important that Apache Ignite is not just a step of ETL pipeline between database or data warehouse and TensorFlow. Apache Ignite is a high-grade database itself. Choosing Apache Ignite and TensorFlow you are getting everything you need to work with operational or historical data and, in the same time, an ability to use this data for neural network training and inference.

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
>>> 
>>> dataset = IgniteDataset(cache_name="SQL_PUBLIC_KITTEN_CACHE")
>>> iterator = dataset.make_one_shot_iterator()
>>> next_obj = iterator.get_next()
>>>
>>> with tf.Session() as sess:
>>>   for _ in range(3):
>>>     print(sess.run(next_obj))

{'key': 1, 'val': {'NAME': b'WARM KITTY'}}
{'key': 2, 'val': {'NAME': b'SOFT KITTY'}}
{'key': 3, 'val': {'NAME': b'LITTLE BALL OF FUR'}}
```

### Structured Objects
[Apache Ignite](https://ignite.apache.org/) allows to store any objects you would like to store. These objects can have any hierarchy. Ignite Dataset provides an ability to work with such objects.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> 
>>> dataset = IgniteDataset(cache_name="IMAGES")
>>> iterator = dataset.make_one_shot_iterator()
>>> next_obj = iterator.get_next()
>>>
>>> with tf.Session() as sess:
>>>   print(sess.run(next_obj))

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
 Neural network training and other computations require transformations that can be done as part of  [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) pipeline if you use Ignite Dataset.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> 
>>> dataset = IgniteDataset(cache_name="IMAGES").map(lambda obj: obj['val']['pixels'])
>>> iterator = dataset.make_one_shot_iterator()
>>> next_obj = iterator.get_next()
>>>
>>> with tf.Session() as sess:
>>>   print(sess.run(next_obj))

[0, 0, 0, 0, ..., 0]
```

### Distributed Training

TensorFlow is a machine learning framework that [natively supports](https://www.tensorflow.org/deploy/distributed) distributed neural network training, inference and other computations. The main idea behind the distributed neural network training is an ability to calculate gradients of loss functions (squares of the errors) on every partition of data (in terms of horizontal partitioning) and then sum them to get loss function gradient of the whole dataset. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla[\sum_1^n(y&space;-&space;\hat{y})^2]&space;=&space;\nabla[\sum_1^{n_1}(y&space;-&space;\hat{y})^2]&space;&plus;&space;\nabla[\sum_{n_1}^{n_2}(y&space;-&space;\hat{y})^2]&space;&plus;&space;...&space;&plus;&space;\nabla[\sum_{n_{k-1}}^n(y&space;-&space;\hat{y})^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\nabla[\sum_1^n(y&space;-&space;\hat{y})^2]&space;=&space;\nabla[\sum_1^{n_1}(y&space;-&space;\hat{y})^2]&space;&plus;&space;\nabla[\sum_{n_1}^{n_2}(y&space;-&space;\hat{y})^2]&space;&plus;&space;...&space;&plus;&space;\nabla[\sum_{n_{k-1}}^n(y&space;-&space;\hat{y})^2]" title="\nabla[\sum_1^n(y - \hat{y})^2] = \nabla[\sum_1^{n_1}(y - \hat{y})^2] + \nabla[\sum_{n_1}^{n_2}(y - \hat{y})^2] + ... + \nabla[\sum_{n_{k-1}}^n(y - \hat{y})^2]" /></a>

Utilizing this ability we can calculate gradients on the nodes the data is stored on, reduce them and then finally update model parameters. It allows to avoid data transfers between nodes and thus to avoid network bottleneck.

Apache Ignite uses horizontal partitioning to store data in distributed cluster. When we create Apache Ignite cache (or table in terms of SQL) we can specify the number of partitions the data will be partitioned on. If, for example, Apache Ignite cluster consists of 10 machines and we creates cache with 10 partitions then every machine will maintain approximately one data partition.

Ignite Dataset allows to utilize these two aspects of distributed neural network training (using TensorFlow) and Apache Ignite partitioning. Ignite Dataset is a computation graph operation that might be performed on a remote worker. The remote worker can override Ignite Dataset parameters (such as `host`, `port` or `part`) by setting correstondent environment variables for worker process (such as `IGNITE_DATASET_HOST`, `IGNITE_DATASET_PORT` or `IGNITE_DATASET_PART`). Using this overriding approach we are able to assign specific partition to every worker so that one worker handles one partition and, at the same time, transparently work with single dataset.

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
>>>         device_iterator = dataset.make_one_shot_iterator()
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

### SSL Connection

Your data should not be accessible without any control. Apache Ignite allows to protect data transfer channels by [SSL](https://en.wikipedia.org/wiki/Transport_Layer_Security) and authentification. Ignite Dataset supports both SSL connection with and without authntication. For more information please see [Apache Ignite SSL/TLS](https://apacheignite.readme.io/docs/ssltls) documentation.

```python
>>> import tensorflow as tf
>>> from tensorflow.contrib.ignite import IgniteDataset
>>> 
>>> dataset = IgniteDataset(cache_name="IMAGES", certfile="client.pem", cert_password="password", username="ignite", password="ignite")
>>> ...
```

### Windows Support

Ignite Dataset is fully compatible with Windows, so you can use it as part of TensorFlow on your Windows workstation as well as on Linux/MacOS systems.

## Try it out

The simplest way to try Ignite Dataset out is to run [Docker](https://www.docker.com/) container with Apache Ignite and loaded [MNIST](http://yann.lecun.com/exdb/mnist/) data and then interruct with it using Ignite Dataset. Such container is available on Docker Hub: [dmitrievanthony/ignite-with-mnist](https://hub.docker.com/r/dmitrievanthony/ignite-with-mnist/). You need to start this container on your machine:

```
docker run -it -p 10800:10800 dmitrievanthony/ignite-with-mnist
```

After that you will be able to work with it following way:

![ignite-dataset-mnist](https://s3.amazonaws.com/helloworld23423423ew23/ignite-dataset-mnist.png "Ignite Dataset Mnist")

## Limitations

Presently Ignite Dataset works with assumption that all objects in the cache have the same structure (homogeneous objects) and the cache contains at least one object. Another limitation concerns structured objects, Ignite Dataset does not support UUID, Maps and Object arrays that might be parts of object structures.