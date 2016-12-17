<!-- This file is machine generated: DO NOT EDIT! -->

# Optimization (contrib)
[TOC]

opt: A module containing optimization routines.

## Other Functions and Classes
- - -

### `class tf.contrib.opt.ExternalOptimizerInterface` {#ExternalOptimizerInterface}

Base class for interfaces with external optimization algorithms.

Subclass this and implement `_minimize` in order to wrap a new optimization
algorithm.

`ExternalOptimizerInterface` should not be instantiated directly; instead use
e.g. `ScipyOptimizerInterface`.

- - -

#### `tf.contrib.opt.ExternalOptimizerInterface.__init__(loss, var_list=None, equalities=None, inequalities=None, **optimizer_kwargs)` {#ExternalOptimizerInterface.__init__}

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

#### `tf.contrib.opt.ExternalOptimizerInterface.minimize(session=None, feed_dict=None, fetches=None, step_callback=None, loss_callback=None)` {#ExternalOptimizerInterface.minimize}

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



- - -

### `class tf.contrib.opt.MovingAverageOptimizer` {#MovingAverageOptimizer}

Optimizer wrapper that maintains a moving average of parameters.
- - -

#### `tf.contrib.opt.MovingAverageOptimizer.__init__(opt, average_decay=0.9999, num_updates=None, sequential_update=True)` {#MovingAverageOptimizer.__init__}

Construct a new MovingAverageOptimizer.

##### Args:


*  <b>`opt`</b>: A tf.Optimizer that will be used to compute and apply gradients.
*  <b>`average_decay`</b>: Float.  Decay to use to maintain the moving averages
                 of trained variables.
                 See tf.train.ExponentialMovingAverage for details.
*  <b>`num_updates`</b>: Optional count of number of updates applied to variables.
               See tf.train.ExponentialMovingAverage for details.
*  <b>`sequential_update`</b>: Bool. If False, will compute the moving average at the
                     same time as the model is updated, potentially doing
                     benign data races.
                     If True, will update the moving average after gradient
                     updates.


- - -

#### `tf.contrib.opt.MovingAverageOptimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#MovingAverageOptimizer.apply_gradients}




- - -

#### `tf.contrib.opt.MovingAverageOptimizer.compute_gradients(loss, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)` {#MovingAverageOptimizer.compute_gradients}

Compute gradients of `loss` for the variables in `var_list`.

This is the first part of `minimize()`.  It returns a list
of (gradient, variable) pairs where "gradient" is the gradient
for "variable".  Note that "gradient" can be a `Tensor`, an
`IndexedSlices`, or `None` if there is no gradient for the
given variable.

##### Args:


*  <b>`loss`</b>: A Tensor containing the value to minimize.
*  <b>`var_list`</b>: Optional list of `tf.Variable` to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKey.TRAINABLE_VARIABLES`.
*  <b>`gate_gradients`</b>: How to gate the computation of gradients.  Can be
    `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
*  <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
*  <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
*  <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.

##### Returns:

  A list of (gradient, variable) pairs. Variable is always present, but
  gradient can be `None`.

##### Raises:


*  <b>`TypeError`</b>: If `var_list` contains anything else than `Variable` objects.
*  <b>`ValueError`</b>: If some arguments are invalid.


- - -

#### `tf.contrib.opt.MovingAverageOptimizer.get_name()` {#MovingAverageOptimizer.get_name}




- - -

#### `tf.contrib.opt.MovingAverageOptimizer.get_slot(var, name)` {#MovingAverageOptimizer.get_slot}

Return a slot named `name` created for `var` by the Optimizer.

Some `Optimizer` subclasses use additional variables.  For example
`Momentum` and `Adagrad` use variables to accumulate updates.  This method
gives access to these `Variable` objects if for some reason you need them.

Use `get_slot_names()` to get the list of slot names created by the
`Optimizer`.

##### Args:


*  <b>`var`</b>: A variable passed to `minimize()` or `apply_gradients()`.
*  <b>`name`</b>: A string.

##### Returns:

  The `Variable` for the slot if it was created, `None` otherwise.


- - -

#### `tf.contrib.opt.MovingAverageOptimizer.get_slot_names()` {#MovingAverageOptimizer.get_slot_names}

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

##### Returns:

  A list of strings.


- - -

#### `tf.contrib.opt.MovingAverageOptimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)` {#MovingAverageOptimizer.minimize}

Add operations to minimize `loss` by updating `var_list`.

This method simply combines calls `compute_gradients()` and
`apply_gradients()`. If you want to process the gradient before applying
them call `compute_gradients()` and `apply_gradients()` explicitly instead
of using this function.

##### Args:


*  <b>`loss`</b>: A `Tensor` containing the value to minimize.
*  <b>`global_step`</b>: Optional `Variable` to increment by one after the
    variables have been updated.
*  <b>`var_list`</b>: Optional list of `Variable` objects to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKeys.TRAINABLE_VARIABLES`.
*  <b>`gate_gradients`</b>: How to gate the computation of gradients.  Can be
    `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
*  <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
*  <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
*  <b>`name`</b>: Optional name for the returned operation.
*  <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.

##### Returns:

  An Operation that updates the variables in `var_list`.  If `global_step`
  was not `None`, that operation also increments `global_step`.

##### Raises:


*  <b>`ValueError`</b>: If some of the variables are not `Variable` objects.


- - -

#### `tf.contrib.opt.MovingAverageOptimizer.swapping_saver(var_list=None, name='swapping_saver', **kwargs)` {#MovingAverageOptimizer.swapping_saver}

Create a saver swapping moving averages and variables.

You should use this saver during training.  It will save the moving averages
of the trained parameters under the original parameter names.  For
evaluations or inference you should use a regular saver and it will
automatically use the moving averages for the trained variable.

You must call this function after all variables have been created and after
you have called Optimizer.minimize().

##### Args:


*  <b>`var_list`</b>: List of variables to save, as per `Saver()`.
            If set to None, will save all the variables that have been
            created before this call.
*  <b>`name`</b>: The name of the saver.
*  <b>`**kwargs`</b>: Keyword arguments of `Saver()`.

##### Returns:

  A `tf.Saver` object.

##### Raises:


*  <b>`RuntimeError`</b>: If apply_gradients or minimize has not been called before.



- - -

### `class tf.contrib.opt.ScipyOptimizerInterface` {#ScipyOptimizerInterface}

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



- - -

### `class tf.contrib.opt.VariableClippingOptimizer` {#VariableClippingOptimizer}

Wrapper optimizer that clips the norm of specified variables after update.

This optimizer delegates all aspects of gradient calculation and application
to an underlying optimizer.  After applying gradients, this optimizer then
clips the variable to have a maximum L2 norm along specified dimensions.
NB: this is quite different from clipping the norm of the gradients.

Multiple instances of `VariableClippingOptimizer` may be chained to specify
different max norms for different subsets of variables.

This is more efficient at serving-time than using normalization during
embedding lookup, at the expense of more expensive training and fewer
guarantees about the norms.

- - -

#### `tf.contrib.opt.VariableClippingOptimizer.__init__(opt, vars_to_clip_dims, max_norm, use_locking=False, colocate_clip_ops_with_vars=False, name='VariableClipping')` {#VariableClippingOptimizer.__init__}

Construct a new clip-norm optimizer.

##### Args:


*  <b>`opt`</b>: The actual optimizer that will be used to compute and apply the
    gradients. Must be one of the Optimizer classes.
*  <b>`vars_to_clip_dims`</b>: A dict with keys as Variables and values as lists
    of dimensions along which to compute the L2-norm.  See
    `tf.clip_by_norm` for more details.
*  <b>`max_norm`</b>: The L2-norm to clip to, for all variables specified.
*  <b>`use_locking`</b>: If `True` use locks for clip update operations.
*  <b>`colocate_clip_ops_with_vars`</b>: If `True`, try colocating the clip norm
    ops with the corresponding variable.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "VariableClipping".



#### Other Methods
- - -

#### `tf.contrib.opt.VariableClippingOptimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#VariableClippingOptimizer.apply_gradients}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.compute_gradients(*args, **kwargs)` {#VariableClippingOptimizer.compute_gradients}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.get_slot(*args, **kwargs)` {#VariableClippingOptimizer.get_slot}




- - -

#### `tf.contrib.opt.VariableClippingOptimizer.get_slot_names(*args, **kwargs)` {#VariableClippingOptimizer.get_slot_names}





