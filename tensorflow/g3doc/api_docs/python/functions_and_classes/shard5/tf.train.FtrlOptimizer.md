Optimizer that implements the FTRL algorithm.

See this [paper](
https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).
- - -

#### `tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')` {#FtrlOptimizer.__init__}

Construct a new FTRL optimizer.

##### Args:


*  <b>`learning_rate`</b>: A float value or a constant float `Tensor`.
*  <b>`learning_rate_power`</b>: A float value, must be less or equal to zero.
*  <b>`initial_accumulator_value`</b>: The starting value for accumulators.
    Only positive values are allowed.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Ftrl".

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.train.FtrlOptimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#FtrlOptimizer.apply_gradients}

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

##### Args:


*  <b>`grads_and_vars`</b>: List of (gradient, variable) pairs as returned by
    `compute_gradients()`.
*  <b>`global_step`</b>: Optional `Variable` to increment by one after the
    variables have been updated.
*  <b>`name`</b>: Optional name for the returned operation.  Default to the
    name passed to the `Optimizer` constructor.

##### Returns:

  An `Operation` that applies the specified gradients. If `global_step`
  was not None, that operation also increments `global_step`.

##### Raises:


*  <b>`TypeError`</b>: If `grads_and_vars` is malformed.
*  <b>`ValueError`</b>: If none of the variables have gradients.


- - -

#### `tf.train.FtrlOptimizer.compute_gradients(loss, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)` {#FtrlOptimizer.compute_gradients}

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

#### `tf.train.FtrlOptimizer.get_name()` {#FtrlOptimizer.get_name}




- - -

#### `tf.train.FtrlOptimizer.get_slot(var, name)` {#FtrlOptimizer.get_slot}

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

#### `tf.train.FtrlOptimizer.get_slot_names()` {#FtrlOptimizer.get_slot_names}

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

##### Returns:

  A list of strings.


- - -

#### `tf.train.FtrlOptimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)` {#FtrlOptimizer.minimize}

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


