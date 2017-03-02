# TensorFlow White Papers

This document identifies white papers about TensorFlow.

### Large-Scale Machine Learning on Heterogeneous Distributed Systems

[Access this white paper.](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**Abstract:** TensorFlow is an interface for expressing machine learning
algorithms, and an implementation for executing such algorithms.
A computation expressed using TensorFlow can be
executed with little or no change on a wide variety of heterogeneous
systems, ranging from mobile devices such as phones
and tablets up to large-scale distributed systems of hundreds
of machines and thousands of computational devices such as
GPU cards. The system is flexible and can be used to express
a wide variety of algorithms, including training and inference
algorithms for deep neural network models, and it has been
used for conducting research and for deploying machine learning
systems into production across more than a dozen areas of
computer science and other fields, including speech recognition,
computer vision, robotics, information retrieval, natural
language processing, geographic information extraction, and
computational drug discovery. This paper describes the TensorFlow
interface and an implementation of that interface that
we have built at Google. The TensorFlow API and a reference
implementation were released as an open-source package under
the Apache 2.0 license in November, 2015 and are available at
www.tensorflow.org.


### TensorFlow: A System for Large-Scale Machine Learning

[Access this white paper.](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**Abstract:** TensorFlow is a machine learning system that operates at
large scale and in heterogeneous environments. TensorFlow
uses dataflow graphs to represent computation,
shared state, and the operations that mutate that state. It
maps the nodes of a dataflow graph across many machines
in a cluster, and within a machine across multiple computational
devices, including multicore CPUs, generalpurpose
GPUs, and custom-designed ASICs known as
Tensor Processing Units (TPUs). This architecture gives
flexibility to the application developer: whereas in previous
“parameter server” designs the management of shared
state is built into the system, TensorFlow enables developers
to experiment with novel optimizations and training algorithms.
TensorFlow supports a variety of applications,
with a focus on training and inference on deep neural networks.
Several Google services use TensorFlow in production,
we have released it as an open-source project, and
it has become widely used for machine learning research.
In this paper, we describe the TensorFlow dataflow model
and demonstrate the compelling performance that TensorFlow
achieves for several real-world applications.
