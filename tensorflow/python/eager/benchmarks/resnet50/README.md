Image classification using the ResNet50 model described in
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

Contents:

- `resnet50.py`: Model definition
- `resnet50_test.py`: Sanity unittests and benchmarks for using the model with
  eager execution enabled.
- `resnet50_graph_test.py`: Sanity unittests and benchmarks when using the same
  model code to construct a TensorFlow graph.

# Benchmarks

Using a synthetic data, run:

```
# Using eager execution
python resnet50_test.py --benchmarks=.

# Using graph execution
python resnet50_graph_test.py --benchmarks=.
```

The above uses the model definition included with the TensorFlow pip
package. To build (and run benchmarks) from source:

```
# Using eager execution
bazel run -c opt --config=cuda :resnet50_test -- --benchmarks=.

# Using graph execution
bazel run -c opt --config=cuda :resnet50_graph_test -- --benchmarks=.
```

(Or remove the `--config=cuda` flag for running on CPU instead of GPU).

On October 31, 2017, the benchmarks demonstrated comparable performance
for eager and graph execution of this particular model when using
a single NVIDIA Titan X (Pascal) GPU on a host with an
Intel Xeon E5-1650 CPU @ 3.50GHz and a batch size of 32.

| Benchmark name                           | batch size    | images/second |
| ---------------------------------------  | ------------- | ------------- |
| eager_train_gpu_batch_32_channels_first  |            32 |           171 |
| graph_train_gpu_batch_32_channels_first  |            32 |           172 |
