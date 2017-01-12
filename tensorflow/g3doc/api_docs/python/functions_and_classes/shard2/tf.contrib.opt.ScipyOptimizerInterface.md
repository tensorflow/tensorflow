Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.

Example:

```python
vector = tf.Variable([7., 7.], 'vector')

# Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))

optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})

with tf.Session() as session:
  optimizer.minimize(session)

# The value of vector should now be [0., 0.].
```

Example with constraints:

```python
vector = tf.Variable([7., 7.], 'vector')

# Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))
# Ensure the vector's y component is = 1.
equalities = [vector[1] - 1.]
# Ensure the vector's x component is >= 1.
inequalities = [vector[0] - 1.]

# Our default SciPy optimization algorithm, L-BFGS-B, does not support
# general constraints. Thus we use SLSQP instead.
optimizer = ScipyOptimizerInterface(
    loss, equalities=equalities, inequalities=inequalities, method='SLSQP')

with tf.Session() as session:
  optimizer.minimize(session)

# The value of vector should now be [1., 1.].
```
- - -

#### `tf.contrib.opt.ScipyOptimizerInterface.__init__(loss, var_list=None, equalities=None, inequalities=None, **optimizer_kwargs)` {#ScipyOptimizerInterface.__init__}

Initialize a new interface instance.

##### Args:


*  <b>`loss`</b>: A scalar `Tensor` to be minimized.
*  <b>`var_list`</b>: Optional list of `Variable` objects to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKeys.TRAINABLE_VARIABLES`.
*  <b>`equalities`</b>: Optional list of equality constraint scalar `Tensor`s to be
    held equal to zero.
*  <b>`inequalities`</b>: Optional list of inequality constraint scalar `Tensor`s
    to be kept nonnegative.
*  <b>`**optimizer_kwargs`</b>: Other subclass-specific keyword arguments.


- - -

#### `tf.contrib.opt.ScipyOptimizerInterface.minimize(session=None, feed_dict=None, fetches=None, step_callback=None, loss_callback=None)` {#ScipyOptimizerInterface.minimize}

Minimize a scalar `Tensor`.

Variables subject to optimization are updated in-place at the end of
optimization.

Note that this method does *not* just return a minimization `Op`, unlike
`Optimizer.minimize()`; instead it actually performs minimization by
executing commands to control a `Session`.

##### Args:


*  <b>`session`</b>: A `Session` instance.
*  <b>`feed_dict`</b>: A feed dict to be passed to calls to `session.run`.
*  <b>`fetches`</b>: A list of `Tensor`s to fetch and supply to `loss_callback`
    as positional arguments.
*  <b>`step_callback`</b>: A function to be called at each optimization step;
    arguments are the current values of all optimization variables
    flattened into a single vector.
*  <b>`loss_callback`</b>: A function to be called every time the loss and gradients
    are computed, with evaluated fetches supplied as positional arguments.


