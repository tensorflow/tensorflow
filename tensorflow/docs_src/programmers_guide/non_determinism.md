# Non-determinism in TensorFlow

One of the first tutorials that you will do when you start Deep Learning is the
MNIST tutorial. You have set all possible random seeds and ensured that weight
initialization is the same, but you get different accuracies in every run. The
brief reason for that is TensorFlow sacrifices some determinism for speed.
For example, [floating point additions are not
associative](https://en.wikipedia.org/wiki/Associative_property#Nonassociativity_of_floating_point_calculation).
`(X + Y) + Z` is not the same as `X + (Y + Z)`. In a classic deep learning
problem, there could be many such non-associative additions when gathering the
gradients from multiple samples in a batch. This results in minor discrepancies
in each epoch, which cascade to significant differences in every run. 

This could be problematic from a parallel computing perspective, but in more
recent times, the Deep Learning community seems to be more concerned with the
accuracy on a blind set, rather than the exactness of the weights obtained from
training. This guide provides you with an explanation of how to analyze and
handle this non-determinism, if you so require.

This tutorial will be based on a simple neural network adder. We will use the
lower-level functions as much as possible. What this network tries to do is to
simply tune all the weights such that they become `1.0`. Full code is provided
below. It seems overwhelming at first, but we will walk through it slowly. 

```python
import numpy as np
import tensorflow as tf
from collections import defaultdict

np.random.seed(0)
np.set_printoptions(precision=1)

runs = 7
total_samples = 10000000
batch_size = 10000
epochs = 1
steps = total_samples / batch_size
num_features = 10
lr = 0.01
data_features = np.random.rand(total_samples, num_features)
data_labels = np.sum(data_features, axis=1)

WEIGHTS = 0
BIASES = 1
PREDICTIONS = 2
LOSSES = 3
W_GRADIENTS = 4
B_GRADIENTS = 5

def train(on_gpu, deterministic):
  nested_dict = lambda: defaultdict(nested_dict)
  output_dict = nested_dict()
  for i in xrange(runs):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    with tf.device('/gpu:0' if on_gpu else '/cpu:0'):
      features = tf.placeholder(tf.float32, shape=[batch_size, num_features])
      labels = tf.placeholder(tf.float32, shape=[batch_size])
      w = tf.get_variable('weights', shape=[num_features, 1])
      b = tf.get_variable('bias', shape=[1])
      if deterministic:
        f_matmul_w = tf.matmul(features, w)
        f_matmul_w_temp = tf.concat([f_matmul_w, tf.ones_like(f_matmul_w)],
            axis=1)
        b_temp = tf.stack([tf.ones_like(b), b], axis=0)
        predictions = tf.squeeze(tf.matmul(f_matmul_w_temp, b_temp))
        loss = tf.matmul(
            tf.expand_dims(tf.square((labels - predictions)), 0),
            tf.expand_dims(tf.ones_like(labels), 1))[0] / batch_size
      else:
        predictions = tf.squeeze(tf.matmul(features, w) + b)
        loss = tf.losses.mean_squared_error(predictions, labels)
      gradients = tf.gradients(loss, [w, b])
      weights_op = tf.assign_sub(w, lr * gradients[0])
      bias_op = tf.assign_sub(b, lr * gradients[1])
      train_op = tf.group(weights_op, bias_op)
      init_op = tf.global_variables_initializer()
    if deterministic:
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    else:
      sess = tf.Session()
    sess.run(init_op)
    for j in xrange(epochs):
      for k in xrange(steps):
        s_idx = k * batch_size
        e_idx = s_idx + batch_size
        run_vals = sess.run(
            [ w,
              b,
              predictions,
              loss,
              gradients[0],
              gradients[1],
              train_op],
            feed_dict={ features: data_features[s_idx: e_idx],
                        labels: data_labels[s_idx: e_idx]})
        for l, l_val in enumerate(run_vals):
          output_dict[i][j][k][l] = np.array(l_val)
    sess.close()
  return output_dict

def compute_error_matrix(output_dict, epoch, step, attr):
  arr = []
  for i in xrange(runs):
    arr.append(output_dict[i][epoch][step][attr])
  arr = np.array(arr)
  total_absolute_error = np.zeros((runs, runs))
  for i in xrange(runs):
    for j in xrange(runs):
      if i != j:
        total_absolute_error[i][j] = np.sum(np.absolute(arr[i] - arr[j]))
  print total_absolute_error
```

We start from the top. It is important to use `np.random.seed` to ensure that
every time the code is run, the same `data_features` and `data_labels` are
generated. The next few lines are variables that we define for this small
experiment.

`train()` takes 2 `boolean` values, `on_gpu` and `deterministic`. In each run,
we first call `tf.reset_default_graph()`, followed by `tf.set_random_seed(0)`.
Refer [here](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) for
more information on the graph-level seeds and operation seeds. Next, we use
`tf.device` to scope the graph onto a CPU or a GPU. We define the standard
variables and operations for graphs next. Within the graphs, there is the case
of `deterministic=True` and `deterministic=False`. We will talk about
that later. Following that, we create a `tf.Session()` and then call a
`sess.run` at every step, where we write the `run_vals` to a dictionary to
analyze the values later on. `compute_error_matrix()` is a function that does an
element-wise subtract, absolute and sum of all elements in the `run_vals`.

## CPU Runs

We first verify that CPU runs are deterministic. We set `on_gpu=False`. We set 
`deterministic=False` as well because the code segments when
`deterministic=True` are for the cases where we want it to run on the GPU and
still be deterministic. We run the following:

```python
output_dict = train(False, False)
compute_error_matrix(output_dict, epochs-1, steps-1, PREDICTIONS)
compute_error_matrix(output_dict, epochs-1, steps-1, LOSSES)
compute_error_matrix(output_dict, epochs-1, steps-1, W_GRADIENTS)
compute_error_matrix(output_dict, epochs-1, steps-1, B_GRADIENTS)
```

As expected, we get error matrices with all zeros even on the last epoch and
last step.

```
PREDICTIONS
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]

LOSSES
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]

W_GRADIENTS
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]

B_GRADIENTS
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]
```

## Non-Deterministic GPU Runs

We move on to non-deterministic GPU runs and the code segments that are used are
as follows:

```python
...
predictions = tf.squeeze(tf.matmul(features, w) + b)
loss = tf.losses.mean_squared_error(predictions, labels)
...
sess = tf.Session()
...
```

The non-determinism could happen from any place where the reductions are not
consistent in every run. Note that the following error matrices are at `epoch=0`
and `step=0`. We run the following:

```python
output_dict = train(True, False)
compute_error_matrix(output_dict, 0, 0, PREDICTIONS)
compute_error_matrix(output_dict, 0, 0, LOSSES)
compute_error_matrix(output_dict, 0, 0, W_GRADIENTS)
compute_error_matrix(output_dict, 0, 0, B_GRADIENTS)
```

The output that you will get might look like the following. These outputs are
cherry-picked. But `PREDICTIONS` and `W_GRADIENTS` will never be non-zero. You
can run it a few times to verify this yourself.

```
PREDICTIONS
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]

LOSSES
[[  0.0e+00   3.8e-06   0.0e+00   3.8e-06   3.8e-06   0.0e+00   3.8e-06]
 [  3.8e-06   0.0e+00   3.8e-06   0.0e+00   0.0e+00   3.8e-06   0.0e+00]
 [  0.0e+00   3.8e-06   0.0e+00   3.8e-06   3.8e-06   0.0e+00   3.8e-06]
 [  3.8e-06   0.0e+00   3.8e-06   0.0e+00   0.0e+00   3.8e-06   0.0e+00]
 [  3.8e-06   0.0e+00   3.8e-06   0.0e+00   0.0e+00   3.8e-06   0.0e+00]
 [  0.0e+00   3.8e-06   0.0e+00   3.8e-06   3.8e-06   0.0e+00   3.8e-06]
 [  3.8e-06   0.0e+00   3.8e-06   0.0e+00   0.0e+00   3.8e-06   0.0e+00]]

W_GRADIENTS
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]

B_GRADIENTS
[[  0.0e+00   0.0e+00   9.5e-07   0.0e+00   0.0e+00   9.5e-07   9.5e-07]
 [  0.0e+00   0.0e+00   9.5e-07   0.0e+00   0.0e+00   9.5e-07   9.5e-07]
 [  9.5e-07   9.5e-07   0.0e+00   9.5e-07   9.5e-07   0.0e+00   0.0e+00]
 [  0.0e+00   0.0e+00   9.5e-07   0.0e+00   0.0e+00   9.5e-07   9.5e-07]
 [  0.0e+00   0.0e+00   9.5e-07   0.0e+00   0.0e+00   9.5e-07   9.5e-07]
 [  9.5e-07   9.5e-07   0.0e+00   9.5e-07   9.5e-07   0.0e+00   0.0e+00]
 [  9.5e-07   9.5e-07   0.0e+00   9.5e-07   9.5e-07   0.0e+00   0.0e+00]]
```

`PREDICTIONS` will never be non-zero because `tf.matmul` is known to give
consistent results on the GPU. There seems to be some non-deterministic
operation in `tf.losses.mean_squared_error`. If we dig deeper, you can trace it
from
[here](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/ops/losses/losses_impl.py)
to `mean_squared_error()` then to `compute_weighted_loss()` then to
`_safe_mean()` which then leads you to `reduce_sum`. It is known that
`reduce_sum` in TensorFlow does not guarantee the order of summations. To solve
this, we have to not use `tf.reduce_sum`. The fix is given when
`deterministic=True` and is reproduced below:

```python
loss = tf.matmul(
    tf.expand_dims(tf.square((labels - predictions)), 0),
    tf.expand_dims(tf.ones_like(labels), 1))[0] / batch_size
```

When this fix is applied, `LOSSES` no longer give non-zero values. The strange
thing you might be asking yourself now is why do `W_GRADIENTS` not have any
errors and `B_GRADIENTS` have errors. The easy answer to that would be that
`tf.reduce_sum` is used as well. The fix is given when `deterministic=True` and
is reproduced below:

```python
f_matmul_w = tf.matmul(features, w)
f_matmul_w_temp = tf.concat([f_matmul_w, tf.ones_like(f_matmul_w)],
    axis=1)
b_temp = tf.stack([tf.ones_like(b), b], axis=0)
predictions = tf.squeeze(tf.matmul(f_matmul_w_temp, b_temp))
```

We basically express the entire forward operation in terms of `tf.matmul`.
However, if you run this code without `allow_soft_placement=True`, it will not
run because certain operations cannot be allocated onto a GPU. Hence, we have to
add this: 

```python
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
```

## Deterministic GPU Runs

With all the changes in place, we run the following:

```python
output_dict = train(True, True)
compute_error_matrix(output_dict, epochs-1, steps-1, PREDICTIONS)
compute_error_matrix(output_dict, epochs-1, steps-1, LOSSES)
compute_error_matrix(output_dict, epochs-1, steps-1, W_GRADIENTS)
compute_error_matrix(output_dict, epochs-1, steps-1, B_GRADIENTS)
```

And we get all zeros in the error matrices, even on the very last `epoch` and
`step`. 

## Conclusion

The deterministic GPU run is hardly called a fix because there might be some
performance sacrificed. This is simply a way to achieve determinism but is not
the best solution. The best solution would be to work on an even lower-level,
but that would mean changing potentially many kernels to have determinism. 


