<p align="center">
  <img src="../images/swift_tensorflow_logo.png">
</p>

# Swift for TensorFlow

Welcome to the Swift for TensorFlow development community!

Swift for TensorFlow is a new way to develop machine learning models. It
gives you the power of
[TensorFlow](https://www.tensorflow.org/programmers_guide/eager) directly
integrated into the [Swift programming language](https://swift.org/about).
With Swift, you can write the following imperative code, and Swift
automatically turns it into **a single TensorFlow Graph** and runs it
with the full performance of TensorFlow Sessions on CPU, GPU and
[TPU](https://cloud.google.com/tpu/docs/tpus).

```swift
import TensorFlow

var x = Tensor([[1, 2], [3, 4]])

for i in 1...5 {
  x += x ⊗ x
}

print(x)
```

Swift combines the flexibility of
[Eager Execution](https://www.tensorflow.org/programmers_guide/eager) with the
high performance of [Graphs and Sessions](https://www.tensorflow.org/programmers_guide/graphs).
Behind the scenes, Swift analyzes your Tensor code and automatically builds
graphs for you. Swift also catches type errors and shape mismatches before
running your code, and has [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
built right in. We believe that machine learning tools are so important that
they deserve **a first-class language and a compiler**.

**Note:** Swift for TensorFlow is an early stage research project. It has been
released to enable open source development and is not yet ready for general use
by machine learning developers.

## Open Source

We have released Swift for TensorFlow as an open-source project on GitHub!

Our [documentation repository](https://github.com/tensorflow/swift) contains a
[project overview](https://github.com/tensorflow/swift/blob/master/docs/DesignOverview.md)
and [technical papers](https://github.com/tensorflow/swift/tree/master/docs)
explaining specific areas in depth. There are also instructions for [installing
pre-built packages](https://github.com/tensorflow/swift/blob/master/Installation.md)
(for macOS and Ubuntu) as well as a simple
[usage tutorial](https://github.com/tensorflow/swift/blob/master/Usage.md).

Moving forward, we will use an open design model and all discussions will be
public.

[Sign up here to join the community Google
group](https://groups.google.com/a/tensorflow.org/d/forum/swift), which we will
use for announcements and general discussion.
