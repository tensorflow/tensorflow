Recurrent Neural Network model.

Implements a language modeling network described in
https://www.tensorflow.org/tutorials/recurrent
that is compatible with (and idiomatic for) eager execution.

To run:

- Download and extract the Penn Treebank dataset from
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  ```sh
  tar xvzf simple-examples.tgz -C /tmp
  ```

- Run: `python rnn_ptb.py --data-dir=/tmp/simple-examples/data`


Benchmarks (using synthetic data):

```
# Using eager execution
python rnn_ptb_test.py --benchmarks=.

# Using graph execution
python rnn_ptb_graph_test.py --benchmarks=.
```

The above uses the model definition included with the TensorFlow pip
package. To build (and run benchmarks) from source:


```
# Using eager execution
bazel run -c opt --config=cuda :rnn_ptb_test -- --benchmarks=.

# Using graph execution
bazel run -c opt --config=cuda :rnn_ptb_graph_test -- --benchmarks=.
```

(Or remove the `--config=cuda` flag for running on CPU instead of GPU).

On October 31, 2017, the benchmarks demonstrated slightly better performance
(3-6%) for graph execution over eager execution for this particular model when
using a single NVIDIA Titan X (Pascal) GPU on a host with an Intel Xeon E5-1650
CPU @ 3.50GHz and a batch size of 32.

| Benchmark name                        | examples/second |
| ------------------------------------  | --------------- |
| eager_cudnn_train_large_gpu_batch_20  |             938 |
| graph_cudnn_train_large_gpu_batch_20  |             971 |
| eager_cudnn_train_small_gpu_batch_20  |            2433 |
| graph_cudnn_train_small_gpu_batch_20  |            2585 |

