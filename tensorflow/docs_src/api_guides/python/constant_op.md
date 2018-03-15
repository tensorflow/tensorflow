# Constants, Sequences, and Random Values

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Constant Value Tensors

TensorFlow provides several operations that you can use to generate constants.

*   @{tf.zeros}
*   @{tf.zeros_like}
*   @{tf.ones}
*   @{tf.ones_like}
*   @{tf.fill}
*   @{tf.constant}

## Sequences

*   @{tf.linspace}
*   @{tf.range}

## Random Tensors

TensorFlow has several ops that create random tensors with different
distributions.  The random ops are stateful, and create new random values each
time they are evaluated.

The `seed` keyword argument in these functions acts in conjunction with
the graph-level random seed. Changing either the graph-level seed using
@{tf.set_random_seed} or the
op-level seed will change the underlying seed of these operations. Setting
neither graph-level nor op-level seed, results in a random seed for all
operations.
See @{tf.set_random_seed}
for details on the interaction between operation-level and graph-level random
seeds.

### Examples:

```python
# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

# Each time we run these ops, different results are generated
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))

# Set an op-level seed to generate repeatable sequences across sessions.
norm = tf.random_normal([2, 3], seed=1234)
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))
sess = tf.Session()
print(sess.run(norm))
print(sess.run(norm))
```

Another common use of random values is the initialization of variables. Also see
the @{$variables$Variables How To}.

```python
# Use random uniform values in [0, 1) as the initializer for a variable of shape
# [2, 3]. The default type is float32.
var = tf.Variable(tf.random_uniform([2, 3]), name="var")
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(var))
```

*   @{tf.random_normal}
*   @{tf.truncated_normal}
*   @{tf.random_uniform}
*   @{tf.random_shuffle}
*   @{tf.random_crop}
*   @{tf.multinomial}
*   @{tf.random_gamma}
*   @{tf.set_random_seed}
