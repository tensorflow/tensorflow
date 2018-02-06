# Defining and Running Benchmarks

This guide contains instructions for defining and running a TensorFlow benchmark. These benchmarks store output in [TestResults](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto) format. If these benchmarks are added to TensorFlow github repo, then we will run them daily with our continuous build and display a graph on our dashboard: https://benchmarks-dot-tensorflow-testing.appspot.com/.

[TOC]


## Defining a Benchmark

Defining a TensorFlow benchmark requires extending from `tf.test.Benchmark`
class and calling `self.report_benchmark` method. For example, take a look at the sample benchmark code below:

```python
import time

import tensorflow as tf


# Define a class that extends from tf.test.Benchmark.
class SampleBenchmark(tf.test.Benchmark):

  # Note: benchmark method name must start with `benchmark`.
  def benchmarkSum(self):
    with tf.Session() as sess:
      x = tf.constant(10)
      y = tf.constant(5)
      result = tf.add(x, y)

      iters = 100
      start_time = time.time()
      for _ in range(iters):
        sess.run(result)
      total_wall_time = time.time() - start_time

      # Call report_benchmark to report a metric value.
      self.report_benchmark(
          name="sum_wall_time",
          # This value should always be per iteration.
          wall_time=total_wall_time/iters,
          iters=iters)

if __name__ == "__main__":
  tf.test.main()
```
See the full example for [SampleBenchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/benchmark/).


Key points to note in the example above:

* Benchmark class extends from `tf.test.Benchmark`.
* Each benchmark method should start with `benchmark` prefix.
* Benchmark method calls `report_benchmark` to report the metric value.


## Running with Python

Use the `--benchmarks` flag to run the benchmark with python. A [BenchmarkEntries](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/util/test_log.proto) proto will be printed.

```
python sample_benchmark.py --benchmarks=SampleBenchmark
```

Setting the flag as `--benchmarks=.` or `--benchmarks=all` would work as well.

(Please ensure that Tensorflow is installed to successfully import the package in the line `import tensorflow as tf`. For installation instructions, see [Installing TensorFlow](https://www.tensorflow.org/install/). This step is not necessary when running with bazel.)


## Adding a `bazel` Target

We have a special target called `tf_py_logged_benchmark` for benchmarks defined under TensorFlow github repo. `tf_py_logged_benchmark` should wrap around a regular `py_test` target. Running a `tf_py_logged_benchmark` would print a [TestResults](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto) proto. Defining a `tf_py_logged_benchmark` also lets us run it with TensorFlow continuous build.

First, define a regular `py_test` target. See example below:

```build
py_test(
  name = "sample_benchmark",
  srcs = ["sample_benchmark.py"],
  srcs_version = "PY2AND3",
  deps = [
    "//tensorflow:tensorflow_py",
  ],
)
```

You can run benchmarks in a `py_test` target by passing `--benchmarks` flag. The benchmark should just print out a [BenchmarkEntries](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/util/test_log.proto) proto.

```shell
bazel test :sample_benchmark --test_arg=--benchmarks=all
```


Now, add the `tf_py_logged_benchmark` target (if available). This target would
pass in `--benchmarks=all` to the wrapped `py_test` target and provide a way to store output for our TensorFlow continuous build. `tf_py_logged_benchmark` target should be available in TensorFlow repository.

```build
load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

tf_py_logged_benchmark(
    name = "sample_logged_benchmark",
    target = "//tensorflow/examples/benchmark:sample_benchmark",
)
```

Use the following command to run the benchmark target:

```shell
bazel test :sample_logged_benchmark
```
