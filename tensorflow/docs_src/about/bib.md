# TensorFlow White Papers

This document identifies white papers about TensorFlow.

## Large-Scale Machine Learning on Heterogeneous Distributed Systems

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


### In BibTeX format

If you use TensorFlow in your research and would like to cite the TensorFlow
system, we suggest you cite this whitepaper.

<pre>
@misc{tensorflow2015-whitepaper,
title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},
url={https://www.tensorflow.org/},
note={Software available from tensorflow.org},
author={
    Mart\'{\i}n~Abadi and
    Ashish~Agarwal and
    Paul~Barham and
    Eugene~Brevdo and
    Zhifeng~Chen and
    Craig~Citro and
    Greg~S.~Corrado and
    Andy~Davis and
    Jeffrey~Dean and
    Matthieu~Devin and
    Sanjay~Ghemawat and
    Ian~Goodfellow and
    Andrew~Harp and
    Geoffrey~Irving and
    Michael~Isard and
    Yangqing Jia and
    Rafal~Jozefowicz and
    Lukasz~Kaiser and
    Manjunath~Kudlur and
    Josh~Levenberg and
    Dandelion~Man\'{e} and
    Rajat~Monga and
    Sherry~Moore and
    Derek~Murray and
    Chris~Olah and
    Mike~Schuster and
    Jonathon~Shlens and
    Benoit~Steiner and
    Ilya~Sutskever and
    Kunal~Talwar and
    Paul~Tucker and
    Vincent~Vanhoucke and
    Vijay~Vasudevan and
    Fernanda~Vi\'{e}gas and
    Oriol~Vinyals and
    Pete~Warden and
    Martin~Wattenberg and
    Martin~Wicke and
    Yuan~Yu and
    Xiaoqiang~Zheng},
  year={2015},
}
</pre>

Or in textual form:

<pre>
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
2015. Software available from tensorflow.org.
</pre>



## TensorFlow: A System for Large-Scale Machine Learning

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

