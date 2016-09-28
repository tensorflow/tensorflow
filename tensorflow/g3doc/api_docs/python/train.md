<!-- This file is machine generated: DO NOT EDIT! -->

# Training
[TOC]

This library provides a set of classes and functions that helps train models.

## Optimizers

The Optimizer base class provides methods to compute gradients for a loss and
apply gradients to variables.  A collection of subclasses implement classic
optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one
of the subclasses.

- - -

### `class tf.train.Optimizer` {#Optimizer}

Base class for optimizers.

This class defines the API to add Ops to train a model.  You never use this
class directly, but instead instantiate one of its subclasses such as
`GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

### Usage

```python
# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable
# objects.
opt_op = opt.minimize(cost, var_list=<list of variables>)
```

In the training program you will just have to run the returned Op.

```python
# Execute opt_op to do one step of training:
opt_op.run()
```

### Processing gradients before applying them.

Calling `minimize()` takes care of both computing the gradients and
applying them to the variables.  If you want to process the gradients
before applying them you can instead use the optimizer in three steps:

1.  Compute the gradients with `compute_gradients()`.
2.  Process the gradients as you wish.
3.  Apply the processed gradients with `apply_gradients()`.

Example:

```python
# Create an optimizer.
opt = GradientDescentOptimizer(learning_rate=0.1)

# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(loss, <list of variables>)

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

# Ask the optimizer to apply the capped gradients.
opt.apply_gradients(capped_grads_and_vars)
```

- - -

#### `tf.train.Optimizer.__init__(use_locking, name)` {#Optimizer.__init__}

Create a new Optimizer.

This must be called by the constructors of subclasses.

##### Args:


*  <b>`use_locking`</b>: Bool. If True apply use locks to prevent concurrent updates
    to variables.
*  <b>`name`</b>: A non-empty string.  The name to use for accumulators created
    for the optimizer.

##### Raises:


*  <b>`ValueError`</b>: If name is malformed.



- - -

#### `tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)` {#Optimizer.minimize}

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

#### `tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)` {#Optimizer.compute_gradients}

Compute gradients of `loss` for the variables in `var_list`.

This is the first part of `minimize()`.  It returns a list
of (gradient, variable) pairs where "gradient" is the gradient
for "variable".  Note that "gradient" can be a `Tensor`, an
`IndexedSlices`, or `None` if there is no gradient for the
given variable.

##### Args:


*  <b>`loss`</b>: A Tensor containing the value to minimize.
*  <b>`var_list`</b>: Optional list of tf.Variable to update to minimize
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

  A list of (gradient, variable) pairs.

##### Raises:


*  <b>`TypeError`</b>: If `var_list` contains anything else than `Variable` objects.
*  <b>`ValueError`</b>: If some arguments are invalid.


- - -

#### `tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#Optimizer.apply_gradients}

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



### Gating Gradients

Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
argument that controls the degree of parallelism during the application of
the gradients.

The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.

<b>`GATE_NONE`</b>: Compute and apply gradients in parallel.  This provides
the maximum parallelism in execution, at the cost of some non-reproducibility
in the results.  For example the two gradients of `matmul` depend on the input
values: With `GATE_NONE` one of the gradients could be applied to one of the
inputs _before_ the other gradient is computed resulting in non-reproducible
results.

<b>`GATE_OP`</b>: For each Op, make sure all gradients are computed before
they are used.  This prevents race conditions for Ops that generate gradients
for multiple inputs where the gradients depend on the inputs.

<b>`GATE_GRAPH`</b>: Make sure all gradients for all variables are computed
before any one of them is used.  This provides the least parallelism but can
be useful if you want to process all gradients before applying any of them.

### Slots

Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer`
allocate and manage additional variables associated with the variables to
train.  These are called <i>Slots</i>.  Slots have names and you can ask the
optimizer for the names of the slots that it uses.  Once you have a slot name
you can ask the optimizer for the variable it created to hold the slot value.

This can be useful if you want to log debug a training algorithm, report stats
about the slots, etc.

- - -

#### `tf.train.Optimizer.get_slot_names()` {#Optimizer.get_slot_names}

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

##### Returns:

  A list of strings.


- - -

#### `tf.train.Optimizer.get_slot(var, name)` {#Optimizer.get_slot}

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



#### Other Methods
- - -

#### `tf.train.Optimizer.get_name()` {#Optimizer.get_name}






- - -

### `class tf.train.GradientDescentOptimizer` {#GradientDescentOptimizer}

Optimizer that implements the gradient descent algorithm.

- - -

#### `tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')` {#GradientDescentOptimizer.__init__}

Construct a new gradient descent optimizer.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning
    rate to use.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "GradientDescent".



- - -

### `class tf.train.AdadeltaOptimizer` {#AdadeltaOptimizer}

Optimizer that implements the Adadelta algorithm. 

See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))

- - -

#### `tf.train.AdadeltaOptimizer.__init__(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')` {#AdadeltaOptimizer.__init__}

Construct a new Adadelta optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value. The learning rate.
*  <b>`rho`</b>: A `Tensor` or a floating point value. The decay rate.
*  <b>`epsilon`</b>: A `Tensor` or a floating point value.  A constant epsilon used
           to better conditioning the grad update.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adadelta".



- - -

### `class tf.train.AdagradOptimizer` {#AdagradOptimizer}

Optimizer that implements the Adagrad algorithm.

See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

- - -

#### `tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')` {#AdagradOptimizer.__init__}

Construct a new Adagrad optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`initial_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adagrad".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_accumulator_value` is invalid.



- - -

### `class tf.train.AdagradDAOptimizer` {#AdagradDAOptimizer}

Adagrad Dual Averaging algorithm for sparse linear models.

See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

This optimizer takes care of regularization of unseen features in a mini batch
by updating them when they are seen with a closed form update rule that is
equivalent to having updated them on every mini-batch.

AdagradDA is typically used when there is a need for large sparsity in the
trained model. This optimizer only guarantees sparsity for linear models. Be
careful when using AdagradDA for deep networks as it will require careful
initialization of the gradient accumulators for it to train.

- - -

#### `tf.train.AdagradDAOptimizer.__init__(learning_rate, global_step, initial_gradient_squared_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='AdagradDA')` {#AdagradDAOptimizer.__init__}

Construct a new AdagradDA optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`global_step`</b>: A `Tensor` containing the current training step number.
*  <b>`initial_gradient_squared_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "AdagradDA".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_gradient_squared_accumulator_value` is
  invalid.



- - -

### `class tf.train.MomentumOptimizer` {#MomentumOptimizer}

Optimizer that implements the Momentum algorithm.

- - -

#### `tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)` {#MomentumOptimizer.__init__}

Construct a new Momentum optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`momentum`</b>: A `Tensor` or a floating point value.  The momentum.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Momentum".



- - -

### `class tf.train.AdamOptimizer` {#AdamOptimizer}

Optimizer that implements the Adam algorithm.

See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

- - -

#### `tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')` {#AdamOptimizer.__init__}

Construct a new Adam optimizer.

Initialization:

```
m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)
```

The update rule for `variable` with gradient `g` uses an optimization
described at the end of section2 of the paper:

```
t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
```

The default value of 1e-8 for epsilon might not be a good default in
general. For example, when training an Inception network on ImageNet a
current good choice is 1.0 or 0.1.

Note that in dense implement of this algorithm, m_t, v_t and variable will
update even if g is zero, but in sparse implement, m_t, v_t and variable
will not update in iterations g is zero.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning rate.
*  <b>`beta1`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 1st moment estimates.
*  <b>`beta2`</b>: A float value or a constant float tensor.
    The exponential decay rate for the 2nd moment estimates.
*  <b>`epsilon`</b>: A small constant for numerical stability.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name for the operations created when applying gradients.
    Defaults to "Adam".



- - -

### `class tf.train.FtrlOptimizer` {#FtrlOptimizer}

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

### `class tf.train.RMSPropOptimizer` {#RMSPropOptimizer}

Optimizer that implements the RMSProp algorithm.

See the [paper]
(http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

- - -

#### `tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')` {#RMSPropOptimizer.__init__}

Construct a new RMSProp optimizer.

Note that in dense implement of this algorithm, m_t and v_t will
update even if g is zero, but in sparse implement, m_t and v_t
will not update in iterations g is zero.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning rate.
*  <b>`decay`</b>: Discounting factor for the history/coming gradient
*  <b>`momentum`</b>: A scalar tensor.
*  <b>`epsilon`</b>: Small value to avoid zero denominator.
*  <b>`use_locking`</b>: If True use locks for update operation.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "RMSProp".




## Gradient Computation

TensorFlow provides functions to compute the derivatives for a given
TensorFlow computation graph, adding operations to the graph. The
optimizer classes automatically compute derivatives on your graph, but
creators of new Optimizers or expert users can call the lower-level
functions below.

- - -

### `tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)` {#gradients}

Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`.

`ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
is a list of `Tensor`, holding the gradients received by the
`ys`. The list must be the same length as `ys`.

`gradients()` adds ops to the graph to output the partial
derivatives of `ys` with respect to `xs`.  It returns a list of
`Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
for y in `ys`.

`grad_ys` is a list of tensors of the same length as `ys` that holds
the initial gradients for each y in `ys`.  When `grad_ys` is None,
we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
user can provide their own initial `grad_ys` to compute the
derivatives using a different initial gradient for each y (e.g., if
one wanted to weight the gradient differently for each value in
each y).

##### Args:


*  <b>`ys`</b>: A `Tensor` or list of tensors to be differentiated.
*  <b>`xs`</b>: A `Tensor` or list of tensors to be used for differentiation.
*  <b>`grad_ys`</b>: Optional. A `Tensor` or list of tensors the same size as
    `ys` and holding the gradients computed for each y in `ys`.
*  <b>`name`</b>: Optional name to use for grouping all the gradient ops together.
    defaults to 'gradients'.
*  <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
*  <b>`gate_gradients`</b>: If True, add a tuple around the gradients returned
    for an operations.  This avoids some race conditions.
*  <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Accepted values are constants defined in the class `AggregationMethod`.

##### Returns:

  A list of `sum(dy/dx)` for each x in `xs`.

##### Raises:


*  <b>`LookupError`</b>: if one of the operations between `x` and `y` does not
    have a registered gradient function.
*  <b>`ValueError`</b>: if the arguments are invalid.


- - -

### `class tf.AggregationMethod` {#AggregationMethod}

A class listing aggregation methods used to combine gradients.

Computing partial derivatives can require aggregating gradient
contributions. This class lists the various methods that can
be used to combine gradients in the graph:

*  `ADD_N`: All of the gradient terms are summed as part of one
   operation using the "AddN" op. It has the property that all
   gradients must be ready before any aggregation is performed.
*  `DEFAULT`: The system-chosen default aggregation method.


- - -

### `tf.stop_gradient(input, name=None)` {#stop_gradient}

Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of
its inputs to be taken into account.  Normally, the gradient generator adds ops
to a graph to compute the derivatives of a specified 'loss' by recursively
finding out inputs that contributed to its computation.  If you insert this op
in the graph it inputs are masked from the gradient generator.  They are not
taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need
to pretend that the value was a constant. Some examples include:

*  The *EM* algorithm where the *M-step* should not involve backpropagation
   through the output of the *E-step*.
*  Contrastive divergence training of Boltzmann machines where, when
   differentiating the energy function, the training must not backpropagate
   through the graph that generated the samples from the model.
*  Adversarial training, where no backprop should happen through the adversarial
   example generation process.

##### Args:


*  <b>`input`</b>: A `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.




## Gradient Clipping

TensorFlow provides several operations that you can use to add clipping
functions to your graph. You can use these functions to perform general data
clipping, but they're particularly useful for handling exploding or vanishing
gradients.

- - -

### `tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)` {#clip_by_value}

Clips tensor values to a specified min and max.

Given a tensor `t`, this operation returns a tensor of the same type and
shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
Any values less than `clip_value_min` are set to `clip_value_min`. Any values
greater than `clip_value_max` are set to `clip_value_max`.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_value_min`</b>: A 0-D (scalar) `Tensor`. The minimum value to clip by.
*  <b>`clip_value_max`</b>: A 0-D (scalar) `Tensor`. The maximum value to clip by.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A clipped `Tensor`.


- - -

### `tf.clip_by_norm(t, clip_norm, axes=None, name=None)` {#clip_by_norm}

Clips tensor values to a maximum L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
along the dimensions given in `axes`. Specifically, in the default case
where all dimensions are used for calculation, if the L2-norm of `t` is
already less than or equal to `clip_norm`, then `t` is not modified. If
the L2-norm is greater than `clip_norm`, then this operation returns a
tensor of the same type and shape as `t` with its values set to:

`t * clip_norm / l2norm(t)`

In this case, the L2-norm of the output tensor is `clip_norm`.

As another example, if `t` is a matrix and `axes == [1]`, then each row
of the output will have L2-norm equal to `clip_norm`. If `axes == [0]`
instead, each column of the output will be clipped.

This operation is typically used to clip gradients before applying them with
an optimizer.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
*  <b>`axes`</b>: A 1-D (vector) `Tensor` of type int32 containing the dimensions
    to use for computing the L2-norm. If `None` (the default), uses all
    dimensions.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A clipped `Tensor`.


- - -

### `tf.clip_by_average_norm(t, clip_norm, name=None)` {#clip_by_average_norm}

Clips tensor values to a maximum average L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
normalizes `t` so that its average L2-norm is less than or equal to
`clip_norm`. Specifically, if the average L2-norm is already less than or
equal to `clip_norm`, then `t` is not modified. If the average L2-norm is
greater than `clip_norm`, then this operation returns a tensor of the same
type and shape as `t` with its values set to:

`t * clip_norm / l2norm_avg(t)`

In this case, the average L2-norm of the output tensor is `clip_norm`.

This operation is typically used to clip gradients before applying them with
an optimizer.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A clipped `Tensor`.


- - -

### `tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)` {#clip_by_global_norm}

Clips values of multiple tensors by the ratio of the sum of their norms.

Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
this operation returns a list of clipped tensors `list_clipped`
and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
if you've already computed the global norm for `t_list`, you can specify
the global norm with `use_norm`.

To perform the clipping, the values `t_list[i]` are set to:

    t_list[i] * clip_norm / max(global_norm, clip_norm)

where:

    global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
otherwise they're all shrunk by the global ratio.

Any of the entries of `t_list` that are of type `None` are ignored.

This is the correct way to perform gradient clipping (for example, see
[Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

However, it is slower than `clip_by_norm()` because all the parameters must be
ready before the clipping operation can be performed.

##### Args:


*  <b>`t_list`</b>: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
*  <b>`use_norm`</b>: A 0-D (scalar) `Tensor` of type `float` (optional). The global
    norm to use. If not provided, `global_norm()` is used to compute the norm.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`list_clipped`</b>: A list of `Tensors` of the same type as `list_t`.
*  <b>`global_norm`</b>: A 0-D (scalar) `Tensor` representing the global norm.

##### Raises:


*  <b>`TypeError`</b>: If `t_list` is not a sequence.


- - -

### `tf.global_norm(t_list, name=None)` {#global_norm}

Computes the global norm of multiple tensors.

Given a tuple or list of tensors `t_list`, this operation returns the
global norm of the elements in all tensors in `t_list`. The global norm is
computed as:

`global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

Any entries in `t_list` that are of type None are ignored.

##### Args:


*  <b>`t_list`</b>: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A 0-D (scalar) `Tensor` of type `float`.

##### Raises:


*  <b>`TypeError`</b>: If `t_list` is not a sequence.



## Decaying the learning rate
- - -

### `tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)` {#exponential_decay}

Applies exponential decay to the learning rate.

When training a model, it is often recommended to lower the learning rate as
the training progresses.  This function applies an exponential decay function
to a provided initial learning rate.  It requires a `global_step` value to
compute the decayed learning rate.  You can just pass a TensorFlow variable
that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
```

If the argument `staircase` is `True`, then `global_step / decay_steps` is an
integer division and the decayed learning rate follows a staircase function.

Example: decay every 100000 steps with a base of 0.96:

```python
...
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

##### Args:


*  <b>`learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate.
*  <b>`global_step`</b>: A scalar `int32` or `int64` `Tensor` or a Python number.
    Global step to use for the decay computation.  Must not be negative.
*  <b>`decay_steps`</b>: A scalar `int32` or `int64` `Tensor` or a Python number.
    Must be positive.  See the decay computation above.
*  <b>`decay_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The decay rate.
*  <b>`staircase`</b>: Boolean.  It `True` decay the learning rate at discrete intervals
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to
    'ExponentialDecay'

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.



## Moving Averages

Some training algorithms, such as GradientDescent and Momentum often benefit
from maintaining a moving average of variables during optimization.  Using the
moving averages for evaluations often improve results significantly.

- - -

### `class tf.train.ExponentialMovingAverage` {#ExponentialMovingAverage}

Maintains moving averages of variables by employing an exponential decay.

When training a model, it is often beneficial to maintain moving averages of
the trained parameters.  Evaluations that use averaged parameters sometimes
produce significantly better results than the final trained values.

The `apply()` method adds shadow copies of trained variables and add ops that
maintain a moving average of the trained variables in their shadow copies.
It is used when building the training model.  The ops that maintain moving
averages are typically run after each training step.
The `average()` and `average_name()` methods give access to the shadow
variables and their names.  They are useful when building an evaluation
model, or when restoring a model from a checkpoint file.  They help use the
moving averages in place of the last trained values for evaluations.

The moving averages are computed using exponential decay.  You specify the
decay value when creating the `ExponentialMovingAverage` object.  The shadow
variables are initialized with the same initial values as the trained
variables.  When you run the ops to maintain the moving averages, each
shadow variable is updated with the formula:

  `shadow_variable -= (1 - decay) * (shadow_variable - variable)`

This is mathematically equivalent to the classic formula below, but the use
of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless
updates to the variables:

  `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

Reasonable values for `decay` are close to 1.0, typically in the
multiple-nines range: 0.999, 0.9999, etc.

Example usage when creating a training model:

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)

...train the model by running training_op...
```

There are two ways to use the moving averages for evaluations:

*  Build a model that uses the shadow variables instead of the variables.
   For this, use the `average()` method which returns the shadow variable
   for a given variable.
*  Build a model normally but load the checkpoint files to evaluate by using
   the shadow variable names.  For this use the `average_name()` method.  See
   the [Saver class](../../api_docs/python/train.md#Saver) for more
   information on restoring saved variables.

Example of restoring the shadow variable values:

```python
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```

- - -

#### `tf.train.ExponentialMovingAverage.__init__(decay, num_updates=None, name='ExponentialMovingAverage')` {#ExponentialMovingAverage.__init__}

Creates a new ExponentialMovingAverage object.

The `apply()` method has to be called to create shadow variables and add
ops to maintain moving averages.

The optional `num_updates` parameter allows one to tweak the decay rate
dynamically. .  It is typical to pass the count of training steps, usually
kept in a variable that is incremented at each step, in which case the
decay rate is lower at the start of training.  This makes moving averages
move faster.  If passed, the actual decay rate used is:

  `min(decay, (1 + num_updates) / (10 + num_updates))`

##### Args:


*  <b>`decay`</b>: Float.  The decay to use.
*  <b>`num_updates`</b>: Optional count of number of updates applied to variables.
*  <b>`name`</b>: String. Optional prefix name to use for the name of ops added in
    `apply()`.


- - -

#### `tf.train.ExponentialMovingAverage.apply(var_list=None)` {#ExponentialMovingAverage.apply}

Maintains moving averages of variables.

`var_list` must be a list of `Variable` or `Tensor` objects.  This method
creates shadow variables for all elements of `var_list`.  Shadow variables
for `Variable` objects are initialized to the variable's initial value.
They will be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
For `Tensor` objects, the shadow variables are initialized to 0.

shadow variables are created with `trainable=False` and added to the
`GraphKeys.ALL_VARIABLES` collection.  They will be returned by calls to
`tf.all_variables()`.

Returns an op that updates all shadow variables as described above.

Note that `apply()` can be called multiple times with different lists of
variables.

##### Args:


*  <b>`var_list`</b>: A list of Variable or Tensor objects. The variables
    and Tensors must be of types float16, float32, or float64.

##### Returns:

  An Operation that updates the moving averages.

##### Raises:


*  <b>`TypeError`</b>: If the arguments are not all float16, float32, or float64.
*  <b>`ValueError`</b>: If the moving average of one of the variables is already
    being computed.


- - -

#### `tf.train.ExponentialMovingAverage.average_name(var)` {#ExponentialMovingAverage.average_name}

Returns the name of the `Variable` holding the average for `var`.

The typical scenario for `ExponentialMovingAverage` is to compute moving
averages of variables during training, and restore the variables from the
computed moving averages during evaluations.

To restore variables, you have to know the name of the shadow variables.
That name and the original variable can then be passed to a `Saver()` object
to restore the variable from the moving average value with:
  `saver = tf.train.Saver({ema.average_name(var): var})`

`average_name()` can be called whether or not `apply()` has been called.

##### Args:


*  <b>`var`</b>: A `Variable` object.

##### Returns:

  A string: The name of the variable that will be used or was used
  by the `ExponentialMovingAverage class` to hold the moving average of
  `var`.


- - -

#### `tf.train.ExponentialMovingAverage.average(var)` {#ExponentialMovingAverage.average}

Returns the `Variable` holding the average of `var`.

##### Args:


*  <b>`var`</b>: A `Variable` object.

##### Returns:

  A `Variable` object or `None` if the moving average of `var`
  is not maintained..


- - -

#### `tf.train.ExponentialMovingAverage.variables_to_restore(moving_avg_variables=None)` {#ExponentialMovingAverage.variables_to_restore}

Returns a map of names to `Variables` to restore.

If a variable has a moving average, use the moving average variable name as
the restore name; otherwise, use the variable name.

For example,

```python
  variables_to_restore = ema.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
```

Below is an example of such mapping:

```
  conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
  conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
  global_step: global_step
```

##### Args:


*  <b>`moving_avg_variables`</b>: a list of variables that require to use of the
    moving variable name to be restored. If None, it will default to
    variables.moving_average_variables() + variables.trainable_variables()

##### Returns:

  A map from restore_names to variables. The restore_name can be the
  moving_average version of the variable name if it exist, or the original
  variable name.




## Coordinator and QueueRunner

See [Threading and Queues](../../how_tos/threading_and_queues/index.md)
for how to use threads and queues.  For documentation on the Queue API,
see [Queues](../../api_docs/python/io_ops.md#queues).

- - -

### `class tf.train.Coordinator` {#Coordinator}

A coordinator for threads.

This class implements a simple mechanism to coordinate the termination of a
set of threads.

#### Usage:

```python
# Create a coordinator.
coord = Coordinator()
# Start a number of threads, passing the coordinator to each of them.
...start thread 1...(coord, ...)
...start thread N...(coord, ...)
# Wait for all the threads to terminate.
coord.join(threads)
```

Any of the threads can call `coord.request_stop()` to ask for all the threads
to stop.  To cooperate with the requests, each thread must check for
`coord.should_stop()` on a regular basis.  `coord.should_stop()` returns
`True` as soon as `coord.request_stop()` has been called.

A typical thread running with a coordinator will do something like:

```python
while not coord.should_stop():
  ...do some work...
```

#### Exception handling:

A thread can report an exception to the coordinator as part of the
`should_stop()` call.  The exception will be re-raised from the
`coord.join()` call.

Thread code:

```python
try:
  while not coord.should_stop():
    ...do some work...
except Exception as e:
  coord.request_stop(e)
```

Main code:

```python
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate.
  coord.join(threads)
except Exception as e:
  ...exception that was passed to coord.request_stop()
```

To simplify the thread implementation, the Coordinator provides a
context handler `stop_on_exception()` that automatically requests a stop if
an exception is raised.  Using the context handler the thread code above
can be written as:

```python
with coord.stop_on_exception():
  while not coord.should_stop():
    ...do some work...
```

#### Grace period for stopping:

After a thread has called `coord.request_stop()` the other threads have a
fixed time to stop, this is called the 'stop grace period' and defaults to 2
minutes.  If any of the threads is still alive after the grace period expires
`coord.join()` raises a RuntimeException reporting the laggards.

```python
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate, give them 10s grace period
  coord.join(threads, stop_grace_period_secs=10)
except RuntimeException:
  ...one of the threads took more than 10s to stop after request_stop()
  ...was called.
except Exception:
  ...exception that was passed to coord.request_stop()
```
- - -

#### `tf.train.Coordinator.__init__(clean_stop_exception_types=None)` {#Coordinator.__init__}

Create a new Coordinator.

##### Args:


*  <b>`clean_stop_exception_types`</b>: Optional tuple of Exception types that should
    cause a clean stop of the coordinator. If an exception of one of these
    types is reported to `request_stop(ex)` the coordinator will behave as
    if `request_stop(None)` was called.  Defaults to
    `(tf.errors.OutOfRangeError,)` which is used by input queues to signal
    the end of input. When feeding training data from a Python iterator it
    is common to add `StopIteration` to this list.


- - -

#### `tf.train.Coordinator.clear_stop()` {#Coordinator.clear_stop}

Clears the stop flag.

After this is called, calls to `should_stop()` will return `False`.


- - -

#### `tf.train.Coordinator.join(threads=None, stop_grace_period_secs=120)` {#Coordinator.join}

Wait for threads to terminate.

This call blocks until a set of threads have terminated.  The set of thread
is the union of the threads passed in the `threads` argument and the list
of threads that registered with the coordinator by calling
`Coordinator.register_thread()`.

After the threads stop, if an `exc_info` was passed to `request_stop`, that
exception is re-raised.

Grace period handling: When `request_stop()` is called, threads are given
'stop_grace_period_secs' seconds to terminate.  If any of them is still
alive after that period expires, a `RuntimeError` is raised.  Note that if
an `exc_info` was passed to `request_stop()` then it is raised instead of
that `RuntimeError`.

##### Args:


*  <b>`threads`</b>: List of `threading.Threads`. The started threads to join in
    addition to the registered threads.
*  <b>`stop_grace_period_secs`</b>: Number of seconds given to threads to stop after
    `request_stop()` has been called.

##### Raises:


*  <b>`RuntimeError`</b>: If any thread is still alive after `request_stop()`
    is called and the grace period expires.


- - -

#### `tf.train.Coordinator.joined` {#Coordinator.joined}




- - -

#### `tf.train.Coordinator.register_thread(thread)` {#Coordinator.register_thread}

Register a thread to join.

##### Args:


*  <b>`thread`</b>: A Python thread to join.


- - -

#### `tf.train.Coordinator.request_stop(ex=None)` {#Coordinator.request_stop}

Request that the threads stop.

After this is called, calls to `should_stop()` will return `True`.

Note: If an exception is being passed in, in must be in the context of
handling the exception (i.e. `try: ... except Exception as ex: ...`) and not
a newly created one.

##### Args:


*  <b>`ex`</b>: Optional `Exception`, or Python `exc_info` tuple as returned by
    `sys.exc_info()`.  If this is the first call to `request_stop()` the
    corresponding exception is recorded and re-raised from `join()`.


- - -

#### `tf.train.Coordinator.should_stop()` {#Coordinator.should_stop}

Check if stop was requested.

##### Returns:

  True if a stop was requested.


- - -

#### `tf.train.Coordinator.stop_on_exception()` {#Coordinator.stop_on_exception}

Context manager to request stop when an Exception is raised.

Code that uses a coordinator must catch exceptions and pass
them to the `request_stop()` method to stop the other threads
managed by the coordinator.

This context handler simplifies the exception handling.
Use it as follows:

```python
with coord.stop_on_exception():
  # Any exception raised in the body of the with
  # clause is reported to the coordinator before terminating
  # the execution of the body.
  ...body...
```

This is completely equivalent to the slightly longer code:

```python
try:
  ...body...
exception Exception as ex:
  coord.request_stop(ex)
```

##### Yields:

  nothing.


- - -

#### `tf.train.Coordinator.wait_for_stop(timeout=None)` {#Coordinator.wait_for_stop}

Wait till the Coordinator is told to stop.

##### Args:


*  <b>`timeout`</b>: Float.  Sleep for up to that many seconds waiting for
    should_stop() to become True.

##### Returns:

  True if the Coordinator is told stop, False if the timeout expired.



- - -

### `class tf.train.QueueRunner` {#QueueRunner}

Holds a list of enqueue operations for a queue, each to be run in a thread.

Queues are a convenient TensorFlow mechanism to compute tensors
asynchronously using multiple threads. For example in the canonical 'Input
Reader' setup one set of threads generates filenames in a queue; a second set
of threads read records from the files, processes them, and enqueues tensors
on a second queue; a third set of threads dequeues these input records to
construct batches and runs them through training operations.

There are several delicate issues when running multiple threads that way:
closing the queues in sequence as the input is exhausted, correctly catching
and reporting exceptions, etc.

The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.
- - -

#### `tf.train.QueueRunner.__init__(queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None, queue_runner_def=None)` {#QueueRunner.__init__}

Create a QueueRunner.

On construction the `QueueRunner` adds an op to close the queue.  That op
will be run if the enqueue ops raise exceptions.

When you later call the `create_threads()` method, the `QueueRunner` will
create one thread for each op in `enqueue_ops`.  Each thread will run its
enqueue op in parallel with the other threads.  The enqueue ops do not have
to all be the same op, but it is expected that they all enqueue tensors in
`queue`.

##### Args:


*  <b>`queue`</b>: A `Queue`.
*  <b>`enqueue_ops`</b>: List of enqueue ops to run in threads later.
*  <b>`close_op`</b>: Op to close the queue. Pending enqueue ops are preserved.
*  <b>`cancel_op`</b>: Op to close the queue and cancel pending enqueue ops.
*  <b>`queue_closed_exception_types`</b>: Optional tuple of Exception types that
    indicate that the queue has been closed when raised during an enqueue
    operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
    case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
    when some of the enqueue ops may dequeue from other Queues.
*  <b>`queue_runner_def`</b>: Optional `QueueRunnerDef` protocol buffer. If specified,
    recreates the QueueRunner from its contents. `queue_runner_def` and the
    other arguments are mutually exclusive.

##### Raises:


*  <b>`ValueError`</b>: If both `queue_runner_def` and `queue` are both specified.
*  <b>`ValueError`</b>: If `queue` or `enqueue_ops` are not provided when not
    restoring from `queue_runner_def`.


- - -

#### `tf.train.QueueRunner.cancel_op` {#QueueRunner.cancel_op}




- - -

#### `tf.train.QueueRunner.close_op` {#QueueRunner.close_op}




- - -

#### `tf.train.QueueRunner.create_threads(sess, coord=None, daemon=False, start=False)` {#QueueRunner.create_threads}

Create threads to run the enqueue ops.

This method requires a session in which the graph was launched.  It creates
a list of threads, optionally starting them.  There is one thread for each
op passed in `enqueue_ops`.

The `coord` argument is an optional coordinator, that the threads will use
to terminate together and report exceptions.  If a coordinator is given,
this method starts an additional thread to close the queue when the
coordinator requests a stop.

This method may be called again as long as all threads from a previous call
have stopped.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`coord`</b>: Optional `Coordinator` object for reporting errors and checking
    stop conditions.
*  <b>`daemon`</b>: Boolean.  If `True` make the threads daemon threads.
*  <b>`start`</b>: Boolean.  If `True` starts the threads.  If `False` the
    caller must call the `start()` method of the returned threads.

##### Returns:

  A list of threads.

##### Raises:


*  <b>`RuntimeError`</b>: If threads from a previous call to `create_threads()` are
  still running.


- - -

#### `tf.train.QueueRunner.enqueue_ops` {#QueueRunner.enqueue_ops}




- - -

#### `tf.train.QueueRunner.exceptions_raised` {#QueueRunner.exceptions_raised}

Exceptions raised but not handled by the `QueueRunner` threads.

Exceptions raised in queue runner threads are handled in one of two ways
depending on whether or not a `Coordinator` was passed to
`create_threads()`:

* With a `Coordinator`, exceptions are reported to the coordinator and
  forgotten by the `QueueRunner`.
* Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
  made available in this `exceptions_raised` property.

##### Returns:

  A list of Python `Exception` objects.  The list is empty if no exception
  was captured.  (No exceptions are captured when using a Coordinator.)


- - -

#### `tf.train.QueueRunner.from_proto(queue_runner_def)` {#QueueRunner.from_proto}

Returns a `QueueRunner` object created from `queue_runner_def`.


- - -

#### `tf.train.QueueRunner.name` {#QueueRunner.name}

The string name of the underlying Queue.


- - -

#### `tf.train.QueueRunner.queue` {#QueueRunner.queue}




- - -

#### `tf.train.QueueRunner.queue_closed_exception_types` {#QueueRunner.queue_closed_exception_types}




- - -

#### `tf.train.QueueRunner.to_proto()` {#QueueRunner.to_proto}

Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

##### Returns:

  A `QueueRunnerDef` protocol buffer.



- - -

### `tf.train.add_queue_runner(qr, collection='queue_runners')` {#add_queue_runner}

Adds a `QueueRunner` to a collection in the graph.

When building a complex model that uses many queues it is often difficult to
gather all the queue runners that need to be run.  This convenience function
allows you to add a queue runner to a well known collection in the graph.

The companion method `start_queue_runners()` can be used to start threads for
all the collected queue runners.

##### Args:


*  <b>`qr`</b>: A `QueueRunner`.
*  <b>`collection`</b>: A `GraphKey` specifying the graph collection to add
    the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.


- - -

### `tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')` {#start_queue_runners}

Starts all queue runners collected in the graph.

This is a companion method to `add_queue_runner()`.  It just starts
threads for all queue runners collected in the graph.  It returns
the list of all threads.

##### Args:


*  <b>`sess`</b>: `Session` used to run the queue ops.  Defaults to the
    default session.
*  <b>`coord`</b>: Optional `Coordinator` for coordinating the started threads.
*  <b>`daemon`</b>: Whether the threads should be marked as `daemons`, meaning
    they don't block program exit.
*  <b>`start`</b>: Set to `False` to only create the threads, not start them.
*  <b>`collection`</b>: A `GraphKey` specifying the graph collection to
    get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  A list of threads.



## Distributed execution

See [Distributed TensorFlow](../../how_tos/distributed/index.md) for
more information about how to configure a distributed TensorFlow program.

- - -

### `class tf.train.Server` {#Server}

An in-process TensorFlow server, for use in distributed training.

A `tf.train.Server` instance encapsulates a set of devices and a
[`tf.Session`](../../api_docs/python/client.md#Session) target that
can participate in distributed training. A server belongs to a
cluster (specified by a [`tf.train.ClusterSpec`](#ClusterSpec)), and
corresponds to a particular task in a named job. The server can
communicate with any other server in the same cluster.

- - -

#### `tf.train.Server.__init__(server_or_cluster_def, job_name=None, task_index=None, protocol=None, config=None, start=True)` {#Server.__init__}

Creates a new server with the given definition.

The `job_name`, `task_index`, and `protocol` arguments are optional, and
override any information provided in `server_or_cluster_def`.

##### Args:


*  <b>`server_or_cluster_def`</b>: A `tf.train.ServerDef` or
    `tf.train.ClusterDef` protocol buffer, or a
    `tf.train.ClusterSpec` object, describing the server to be
    created and/or the cluster of which it is a member.
*  <b>`job_name`</b>: (Optional.) Specifies the name of the job of which the server
    is a member. Defaults to the value in `server_or_cluster_def`, if
    specified.
*  <b>`task_index`</b>: (Optional.) Specifies the task index of the server in its
    job. Defaults to the value in `server_or_cluster_def`, if specified.
    Otherwise defaults to 0 if the server's job has only one task.
*  <b>`protocol`</b>: (Optional.) Specifies the protocol to be used by the server.
    Acceptable values include `"grpc"`. Defaults to the value in
    `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
*  <b>`config`</b>: (Options.) A `tf.ConfigProto` that specifies default
    configuration options for all sessions that run on this server.
*  <b>`start`</b>: (Optional.) Boolean, indicating whether to start the server
    after creating it. Defaults to `True`.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    creating the TensorFlow server.


- - -

#### `tf.train.Server.create_local_server(config=None, start=True)` {#Server.create_local_server}

Creates a new single-process cluster running on the local host.

This method is a convenience wrapper for creating a
`tf.train.Server` with a `tf.train.ServerDef` that specifies a
single-process cluster containing a single task in a job called
`"local"`.

##### Args:


*  <b>`config`</b>: (Options.) A `tf.ConfigProto` that specifies default
    configuration options for all sessions that run on this server.
*  <b>`start`</b>: (Optional.) Boolean, indicating whether to start the server after
    creating it. Defaults to `True`.

##### Returns:

  A local `tf.train.Server`.


- - -

#### `tf.train.Server.target` {#Server.target}

Returns the target for a `tf.Session` to connect to this server.

To create a
[`tf.Session`](../../api_docs/python/client.md#Session) that
connects to this server, use the following snippet:

```python
server = tf.train.Server(...)
with tf.Session(server.target):
  # ...
```

##### Returns:

  A string containing a session target for this server.


- - -

#### `tf.train.Server.server_def` {#Server.server_def}

Returns the `tf.train.ServerDef` for this server.

##### Returns:

  A `tf.train.ServerDef` protocol buffer that describes the configuration
  of this server.



- - -

#### `tf.train.Server.start()` {#Server.start}

Starts this server.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    starting the TensorFlow server.


- - -

#### `tf.train.Server.join()` {#Server.join}

Blocks until the server has shut down.

This method currently blocks forever.

##### Raises:

  tf.errors.OpError: Or one of its subclasses if an error occurs while
    joining the TensorFlow server.



- - -

### `class tf.train.Supervisor` {#Supervisor}

A training helper that checkpoints models and computes summaries.

The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
and a `SessionManager` that takes care of common needs of TensorFlow
training programs.

#### Use for a single program

```python
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a TensorFlow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

Within the `with sv.managed_session()` block all variables in the graph have
been initialized.  In addition, a few services have been started to
checkpoint the model and add summaries to the event log.

If the program crashes and is restarted, the managed session automatically
reinitialize variables from the most recent checkpoint.

The supervisor is notified of any exception raised by one of the services.
After an exception is raised, `should_stop()` returns `True`.  In that case
the training loop should also stop.  This is why the training loop has to
check for `sv.should_stop()`.

Exceptions that indicate that the training inputs have been exhausted,
`tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
but are not re-raised from the `with` block: they indicate a normal
termination.

#### Use for multiple replicas

To train with replicas you deploy the same program in a `Cluster`.
One of the tasks must be identified as the *chief*: the task that handles
initialization, checkpoints, summaries, and recovery.  The other tasks
depend on the *chief* for these services.

The only change you have to do to the single program code is to indicate
if the program is running as the *chief*.

```python
# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

In the *chief* task, the `Supervisor` works exactly as in the first example
above.  In the other tasks `sv.managed_session()` waits for the Model to have
been initialized before returning a session to the training code.  The
non-chief tasks depend on the chief task for initializing the model.

If one of the tasks crashes and restarts, `managed_session()`
checks if the Model is initialized.  If yes, it just creates a session and
returns it to the training code that proceeds normally.  If the model needs
to be initialized, the chief task takes care of reinitializing it; the other
tasks just wait for the model to have been initialized.

NOTE: This modified program still works fine as a single program.
The single program marks itself as the chief.

#### What `master` string to use

Whether you are running on your machine or in the cluster you can use the
following values for the --master flag:

* Specifying `''` requests an in-process session that does not use RPC.

* Specifying `'local'` requests a session that uses the RPC-based
  "Master interface" to run TensorFlow programs. See
  [`tf.train.Server.create_local_server()`](#Server.create_local_server) for
  details.

* Specifying `'grpc://hostname:port'` requests a session that uses
  the RPC interface to a specific , and also allows the in-process
  master to access remote tensorflow workers. Often, it is
  appropriate to pass `server.target` (for some `tf.train.Server`
  named `server).

#### Advanced use

##### Launching additional services

`managed_session()` launches the Checkpoint and Summary services (threads).
If you need more services to run you can simply launch them in the block
controlled by `managed_session()`.

Example: Start a thread to print losses.  We want this thread to run
every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

##### Launching fewer services

`managed_session()` launches the "summary" and "checkpoint" threads which use
either the optionally `summary_op` and `saver` passed to the constructor, or
default ones created automatically by the supervisor.  If you want to run
your own summary and checkpointing logic, disable these services by passing
`None` to the `summary_op` and `saver` parameters.

Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

##### Custom model initialization

`managed_session()` only supports initializing the model by running an
`init_op` or restoring from the latest checkpoint.  If you have special
initialization needs, see how to specify a `local_init_op` when creating the
supervisor.  You can also use the `SessionManager` directly to create a
session and check if it could be initialized automatically.

- - -

#### `tf.train.Supervisor.__init__(graph=None, ready_op=0, ready_for_local_init_op=0, is_chief=True, init_op=0, init_feed_dict=None, local_init_op=0, logdir=None, summary_op=0, saver=0, global_step=0, save_summaries_secs=120, save_model_secs=600, recovery_wait_secs=30, stop_grace_secs=120, checkpoint_basename='model.ckpt', session_manager=None, summary_writer=0, init_fn=None)` {#Supervisor.__init__}

Create a `Supervisor`.

##### Args:


*  <b>`graph`</b>: A `Graph`.  The graph that the model will use.  Defaults to the
    default `Graph`.  The supervisor may add operations to the graph before
    creating a session, but the graph should not be modified by the caller
    after passing it to the supervisor.
*  <b>`ready_op`</b>: 1-D string `Tensor`.  This tensor is evaluated by supervisors in
    `prepare_or_wait_for_session()` to check if the model is ready to use.
    The model is considered ready if it returns an empty array.  Defaults to
    the tensor returned from `tf.report_uninitialized_variables()`  If
    `None`, the model is not checked for readiness.
*  <b>`ready_for_local_init_op`</b>: 1-D string `Tensor`.  This tensor is evaluated by
    supervisors in `prepare_or_wait_for_session()` to check if the model is
    ready to run the local_init_op.
    The model is considered ready if it returns an empty array.  Defaults to
    the tensor returned from
    `tf.report_uninitialized_variables(tf.all_variables())`. If `None`, the
    model is not checked for readiness before running local_init_op.
*  <b>`is_chief`</b>: If True, create a chief supervisor in charge of initializing
    and restoring the model.  If False, create a supervisor that relies
    on a chief supervisor for inits and restore.
*  <b>`init_op`</b>: `Operation`.  Used by chief supervisors to initialize the model
    when it can not be recovered.  Defaults to an `Operation` that
    initializes all variables.  If `None`, no initialization is done
    automatically unless you pass a value for `init_fn`, see below.
*  <b>`init_feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
*  <b>`local_init_op`</b>: `Operation`. Used by all supervisors to run initializations
    that should run for every new supervisor instance. By default these
    are table initializers and initializers for local variables.
    If `None`, no further per supervisor-instance initialization is
    done automatically.
*  <b>`logdir`</b>: A string.  Optional path to a directory where to checkpoint the
    model and log events for the visualizer.  Used by chief supervisors.
    The directory will be created if it does not exist.
*  <b>`summary_op`</b>: An `Operation` that returns a Summary for the event logs.
    Used by chief supervisors if a `logdir` was specified.  Defaults to the
    operation returned from merge_all_summaries().  If `None`, summaries are
    not computed automatically.
*  <b>`saver`</b>: A Saver object.  Used by chief supervisors if a `logdir` was
    specified.  Defaults to the saved returned by Saver().
    If `None`, the model is not saved automatically.
*  <b>`global_step`</b>: An integer Tensor of size 1 that counts steps.  The value
    from 'global_step' is used in summaries and checkpoint filenames.
    Default to the op named 'global_step' in the graph if it exists, is of
    rank 1, size 1, and of type tf.int32 ot tf.int64.  If `None` the global
    step is not recorded in summaries and checkpoint files.  Used by chief
    supervisors if a `logdir` was specified.
*  <b>`save_summaries_secs`</b>: Number of seconds between the computation of
    summaries for the event log.  Defaults to 120 seconds.  Pass 0 to
    disable summaries.
*  <b>`save_model_secs`</b>: Number of seconds between the creation of model
    checkpoints.  Defaults to 600 seconds.  Pass 0 to disable checkpoints.
*  <b>`recovery_wait_secs`</b>: Number of seconds between checks that the model
    is ready.  Used by supervisors when waiting for a chief supervisor
    to initialize or restore the model.  Defaults to 30 seconds.
*  <b>`stop_grace_secs`</b>: Grace period, in seconds, given to running threads to
    stop when `stop()` is called.  Defaults to 120 seconds.
*  <b>`checkpoint_basename`</b>: The basename for checkpoint saving.
*  <b>`session_manager`</b>: `SessionManager`, which manages Session creation and
    recovery. If it is `None`, a default `SessionManager` will be created
    with the set of arguments passed in for backwards compatibility.
*  <b>`summary_writer`</b>: `SummaryWriter` to use or `USE_DEFAULT`.  Can be `None`
    to indicate that no summaries should be written.
*  <b>`init_fn`</b>: Optional callable used to initialize the model. Called
    after the optional `init_op` is called.  The callable must accept one
    argument, the session being initialized.

##### Returns:

  A `Supervisor`.


- - -

#### `tf.train.Supervisor.managed_session(master='', config=None, start_standard_services=True, close_summary_writer=True)` {#Supervisor.managed_session}

Returns a context manager for a managed session.

This context manager creates and automatically recovers a session.  It
optionally starts the standard services that handle checkpoints and
summaries.  It monitors exceptions raised from the `with` block or from the
services and stops the supervisor as needed.

The context manager is typically used as follows:

```python
def train():
  sv = tf.train.Supervisor(...)
  with sv.managed_session(<master>) as sess:
    for step in xrange(..):
      if sv.should_stop():
        break
      sess.run(<my training op>)
      ...do other things needed at each training step...
```

An exception raised from the `with` block or one of the service threads is
raised again when the block exits.  This is done after stopping all threads
and closing the session.  For example, an `AbortedError` exception, raised
in case of preemption of one of the workers in a distributed model, is
raised again when the block exits.

If you want to retry the training loop in case of preemption you can do it
as follows:

```python
def main(...):
  while True
    try:
      train()
    except tf.errors.Aborted:
      pass
```

As a special case, exceptions used for control flow, such as
`OutOfRangeError` which reports that input queues are exhausted, are not
raised again from the `with` block: they indicate a clean termination of
the training loop and are considered normal termination.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.
    Passed as-is to create the session.
*  <b>`start_standard_services`</b>: Whether to start the standard services,
    such as checkpoint, summary and step counter.
*  <b>`close_summary_writer`</b>: Whether to close the summary writer when
    closing the session.  Defaults to True.

##### Returns:

  A context manager that yields a `Session` restored from the latest
  checkpoint or initialized from scratch if not checkpoint exists.  The
  session is closed when the `with` block exits.


- - -

#### `tf.train.Supervisor.prepare_or_wait_for_session(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` {#Supervisor.prepare_or_wait_for_session}

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as
needed, or wait for a session to be ready.  If running as the chief
and `start_standard_service` is set to True, also call the session
manager to start the standard services.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session,
    which is passed as-is to create the session.
*  <b>`wait_for_checkpoint`</b>: Whether we should wait for the availability of a
    checkpoint before creating Session. Defaults to False.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.
*  <b>`start_standard_services`</b>: Whether to start the standard services and the
    queue runners.

##### Returns:

  A Session object that can be used to drive the model.


- - -

#### `tf.train.Supervisor.start_standard_services(sess)` {#Supervisor.start_standard_services}

Start the standard services for 'sess'.

This starts services in the background.  The services started depend
on the parameters to the constructor and may include:

  - A Summary thread computing summaries every save_summaries_secs.
  - A Checkpoint thread saving the model every save_model_secs.
  - A StepCounter thread measure step time.

##### Args:


*  <b>`sess`</b>: A Session.

##### Returns:

  A list of threads that are running the standard services.  You can use
  the Supervisor's Coordinator to join these threads with:
    sv.coord.Join(<list of threads>)

##### Raises:


*  <b>`RuntimeError`</b>: If called with a non-chief Supervisor.
*  <b>`ValueError`</b>: If not `logdir` was passed to the constructor as the
    services need a log directory.


- - -

#### `tf.train.Supervisor.start_queue_runners(sess, queue_runners=None)` {#Supervisor.start_queue_runners}

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
are already started automatically when you create a session with the
supervisor, so unless you have non-collected queue runners to start
you do not need to call this explicitly.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`queue_runners`</b>: A list of `QueueRunners`. If not specified, we'll use the
    list of queue runners gathered in the graph under the key
    `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  The list of threads started for the `QueueRunners`.


- - -

#### `tf.train.Supervisor.summary_computed(sess, summary, global_step=None)` {#Supervisor.summary_computed}

Indicate that a summary was computed.

##### Args:


*  <b>`sess`</b>: A `Session` object.
*  <b>`summary`</b>: A Summary proto, or a string holding a serialized summary proto.
*  <b>`global_step`</b>: Int. global step this summary is associated with. If `None`,
    it will try to fetch the current step.

##### Raises:


*  <b>`TypeError`</b>: if 'summary' is not a Summary proto or a string.
*  <b>`RuntimeError`</b>: if the Supervisor was created without a `logdir`.



- - -

#### `tf.train.Supervisor.stop(threads=None, close_summary_writer=True)` {#Supervisor.stop}

Stop the services and the coordinator.

This does not close the session.

##### Args:


*  <b>`threads`</b>: Optional list of threads to join with the coordinator.  If
    `None`, defaults to the threads running the standard services, the
    threads started for `QueueRunners`, and the threads started by the
    `loop()` method.  To wait on additional threads, pass the
    list in this parameter.
*  <b>`close_summary_writer`</b>: Whether to close the `summary_writer`.  Defaults to
    `True` if the summary writer was created by the supervisor, `False`
    otherwise.


- - -

#### `tf.train.Supervisor.request_stop(ex=None)` {#Supervisor.request_stop}

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

##### Args:


*  <b>`ex`</b>: Optional `Exception`, or Python `exc_info` tuple as returned by
    `sys.exc_info()`.  If this is the first call to `request_stop()` the
    corresponding exception is recorded and re-raised from `join()`.


- - -

#### `tf.train.Supervisor.should_stop()` {#Supervisor.should_stop}

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

##### Returns:

  True if the coordinator was told to stop, False otherwise.


- - -

#### `tf.train.Supervisor.stop_on_exception()` {#Supervisor.stop_on_exception}

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

##### Returns:

  A context handler.


- - -

#### `tf.train.Supervisor.wait_for_stop()` {#Supervisor.wait_for_stop}

Block waiting for the coordinator to stop.



#### Other Methods
- - -

#### `tf.train.Supervisor.Loop(timer_interval_secs, target, args=None, kwargs=None)` {#Supervisor.Loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
repeatedly.  Otherwise it calls it every `timer_interval_secs`
seconds.  The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor
so it does not need to be passed to the `stop()` method.

##### Args:


*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.


- - -

#### `tf.train.Supervisor.PrepareSession(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` {#Supervisor.PrepareSession}

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as
needed, or wait for a session to be ready.  If running as the chief
and `start_standard_service` is set to True, also call the session
manager to start the standard services.

##### Args:


*  <b>`master`</b>: name of the TensorFlow master to use.  See the `tf.Session`
    constructor for how this is interpreted.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session,
    which is passed as-is to create the session.
*  <b>`wait_for_checkpoint`</b>: Whether we should wait for the availability of a
    checkpoint before creating Session. Defaults to False.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.
*  <b>`start_standard_services`</b>: Whether to start the standard services and the
    queue runners.

##### Returns:

  A Session object that can be used to drive the model.


- - -

#### `tf.train.Supervisor.RequestStop(ex=None)` {#Supervisor.RequestStop}

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

##### Args:


*  <b>`ex`</b>: Optional `Exception`, or Python `exc_info` tuple as returned by
    `sys.exc_info()`.  If this is the first call to `request_stop()` the
    corresponding exception is recorded and re-raised from `join()`.


- - -

#### `tf.train.Supervisor.ShouldStop()` {#Supervisor.ShouldStop}

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

##### Returns:

  True if the coordinator was told to stop, False otherwise.


- - -

#### `tf.train.Supervisor.StartQueueRunners(sess, queue_runners=None)` {#Supervisor.StartQueueRunners}

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
are already started automatically when you create a session with the
supervisor, so unless you have non-collected queue runners to start
you do not need to call this explicitly.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`queue_runners`</b>: A list of `QueueRunners`. If not specified, we'll use the
    list of queue runners gathered in the graph under the key
    `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  The list of threads started for the `QueueRunners`.


- - -

#### `tf.train.Supervisor.StartStandardServices(sess)` {#Supervisor.StartStandardServices}

Start the standard services for 'sess'.

This starts services in the background.  The services started depend
on the parameters to the constructor and may include:

  - A Summary thread computing summaries every save_summaries_secs.
  - A Checkpoint thread saving the model every save_model_secs.
  - A StepCounter thread measure step time.

##### Args:


*  <b>`sess`</b>: A Session.

##### Returns:

  A list of threads that are running the standard services.  You can use
  the Supervisor's Coordinator to join these threads with:
    sv.coord.Join(<list of threads>)

##### Raises:


*  <b>`RuntimeError`</b>: If called with a non-chief Supervisor.
*  <b>`ValueError`</b>: If not `logdir` was passed to the constructor as the
    services need a log directory.


- - -

#### `tf.train.Supervisor.Stop(threads=None, close_summary_writer=True)` {#Supervisor.Stop}

Stop the services and the coordinator.

This does not close the session.

##### Args:


*  <b>`threads`</b>: Optional list of threads to join with the coordinator.  If
    `None`, defaults to the threads running the standard services, the
    threads started for `QueueRunners`, and the threads started by the
    `loop()` method.  To wait on additional threads, pass the
    list in this parameter.
*  <b>`close_summary_writer`</b>: Whether to close the `summary_writer`.  Defaults to
    `True` if the summary writer was created by the supervisor, `False`
    otherwise.


- - -

#### `tf.train.Supervisor.StopOnException()` {#Supervisor.StopOnException}

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

##### Returns:

  A context handler.


- - -

#### `tf.train.Supervisor.SummaryComputed(sess, summary, global_step=None)` {#Supervisor.SummaryComputed}

Indicate that a summary was computed.

##### Args:


*  <b>`sess`</b>: A `Session` object.
*  <b>`summary`</b>: A Summary proto, or a string holding a serialized summary proto.
*  <b>`global_step`</b>: Int. global step this summary is associated with. If `None`,
    it will try to fetch the current step.

##### Raises:


*  <b>`TypeError`</b>: if 'summary' is not a Summary proto or a string.
*  <b>`RuntimeError`</b>: if the Supervisor was created without a `logdir`.


- - -

#### `tf.train.Supervisor.WaitForStop()` {#Supervisor.WaitForStop}

Block waiting for the coordinator to stop.


- - -

#### `tf.train.Supervisor.coord` {#Supervisor.coord}

Return the Coordinator used by the Supervisor.

The Coordinator can be useful if you want to run multiple threads
during your training.

##### Returns:

  A Coordinator object.


- - -

#### `tf.train.Supervisor.global_step` {#Supervisor.global_step}

Return the global_step Tensor used by the supervisor.

##### Returns:

  An integer Tensor for the global_step.


- - -

#### `tf.train.Supervisor.init_feed_dict` {#Supervisor.init_feed_dict}

Return the feed dictionary used when evaluating the `init_op`.

##### Returns:

  A feed dictionary or `None`.


- - -

#### `tf.train.Supervisor.init_op` {#Supervisor.init_op}

Return the Init Op used by the supervisor.

##### Returns:

  An Op or `None`.


- - -

#### `tf.train.Supervisor.is_chief` {#Supervisor.is_chief}

Return True if this is a chief supervisor.

##### Returns:

  A bool.


- - -

#### `tf.train.Supervisor.loop(timer_interval_secs, target, args=None, kwargs=None)` {#Supervisor.loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
repeatedly.  Otherwise it calls it every `timer_interval_secs`
seconds.  The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor
so it does not need to be passed to the `stop()` method.

##### Args:


*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.


- - -

#### `tf.train.Supervisor.ready_for_local_init_op` {#Supervisor.ready_for_local_init_op}




- - -

#### `tf.train.Supervisor.ready_op` {#Supervisor.ready_op}

Return the Ready Op used by the supervisor.

##### Returns:

  An Op or `None`.


- - -

#### `tf.train.Supervisor.save_model_secs` {#Supervisor.save_model_secs}

Return the delay between checkpoints.

##### Returns:

  A timestamp.


- - -

#### `tf.train.Supervisor.save_path` {#Supervisor.save_path}

Return the save path used by the supervisor.

##### Returns:

  A string.


- - -

#### `tf.train.Supervisor.save_summaries_secs` {#Supervisor.save_summaries_secs}

Return the delay between summary computations.

##### Returns:

  A timestamp.


- - -

#### `tf.train.Supervisor.saver` {#Supervisor.saver}

Return the Saver used by the supervisor.

##### Returns:

  A Saver object.


- - -

#### `tf.train.Supervisor.session_manager` {#Supervisor.session_manager}

Return the SessionManager used by the Supervisor.

##### Returns:

  A SessionManager object.


- - -

#### `tf.train.Supervisor.summary_op` {#Supervisor.summary_op}

Return the Summary Tensor used by the chief supervisor.

##### Returns:

  A string Tensor for the summary or `None`.


- - -

#### `tf.train.Supervisor.summary_writer` {#Supervisor.summary_writer}

Return the SummaryWriter used by the chief supervisor.

##### Returns:

  A SummaryWriter.



- - -

### `class tf.train.SessionManager` {#SessionManager}

Training helper that restores from checkpoint and creates session.

This class is a small wrapper that takes care of session creation and
checkpoint recovery. It also provides functions that to facilitate
coordination among multiple training threads or processes.

* Checkpointing trained variables as the training progresses.
* Initializing variables on startup, restoring them from the most recent
  checkpoint after a crash, or wait for checkpoints to become available.

### Usage:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
  sm = SessionManager()
  sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`prepare_session()` initializes or restores a model. It requires `init_op`
and `saver` as an argument.

A second process could wait for the model to be ready by doing the following:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will wait for the model to become ready.
  sm = SessionManager()
  sess = sm.wait_for_session(master)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`wait_for_session()` waits for a model to be initialized by other processes.
- - -

#### `tf.train.SessionManager.__init__(local_init_op=None, ready_op=None, ready_for_local_init_op=None, graph=None, recovery_wait_secs=30)` {#SessionManager.__init__}

Creates a SessionManager.

The `local_init_op` is an `Operation` that is run always after a new session
was created. If `None`, this step is skipped.

The `ready_op` is an `Operation` used to check if the model is ready.  The
model is considered ready if that operation returns an empty string tensor.
If the operation returns non empty string tensor, the elements are
concatenated and used to indicate to the user why the model is not ready.

The `ready_for_local_init_op` is an `Operation` used to check if the model
is ready to run local_init_op.  The model is considered ready if that
operation returns an empty string tensor. If the operation returns non empty
string tensor, the elements are concatenated and used to indicate to the
user why the model is not ready.

If `ready_op` is `None`, the model is not checked for readiness.

`recovery_wait_secs` is the number of seconds between checks that
the model is ready.  It is used by processes to wait for a model to
be initialized or restored.  Defaults to 30 seconds.

##### Args:


*  <b>`local_init_op`</b>: An `Operation` run immediately after session creation.
     Usually used to initialize tables and local variables.
*  <b>`ready_op`</b>: An `Operation` to check if the model is initialized.
*  <b>`ready_for_local_init_op`</b>: An `Operation` to check if the model is ready
     to run local_init_op.
*  <b>`graph`</b>: The `Graph` that the model will use.
*  <b>`recovery_wait_secs`</b>: Seconds between checks for the model to be ready.

##### Raises:


*  <b>`ValueError`</b>: If ready_for_local_init_op is not None but local_init_op is
    None


- - -

#### `tf.train.SessionManager.prepare_session(master, init_op=None, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None)` {#SessionManager.prepare_session}

Creates a `Session`. Makes sure the model is ready to be used.

Creates a `Session` on 'master'. If a `saver` object is passed in, and
`checkpoint_dir` points to a directory containing valid checkpoint
files, then it will try to recover the model from checkpoint. If
no checkpoint files are available, and `wait_for_checkpoint` is
`True`, then the process would check every `recovery_wait_secs`,
up to `max_wait_secs`, for recovery to succeed.

If the model cannot be recovered successfully then it is initialized by
either running the provided `init_op`, or calling the provided `init_fn`.
The local_init_op is also run after init_op and init_fn, regardless of
whether the model was recovered successfully, but only if
ready_for_local_init_op passes.

It is an error if the model cannot be recovered and no `init_op`
or `init_fn` or `local_init_op` are passed.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`init_op`</b>: Optional `Operation` used to initialize the model.
*  <b>`saver`</b>: A `Saver` object used to restore a model.
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files.
*  <b>`wait_for_checkpoint`</b>: Whether to wait for checkpoint to become available.
*  <b>`max_wait_secs`</b>: Maximum time to wait for checkpoints to become available.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.
*  <b>`init_feed_dict`</b>: Optional dictionary that maps `Tensor` objects to feed
    values.  This feed dictionary is passed to the session `run()` call when
    running the init op.
*  <b>`init_fn`</b>: Optional callable used to initialize the model. Called after the
    optional `init_op` is called.  The callable must accept one argument,
    the session being initialized.

##### Returns:

  A `Session` object that can be used to drive the model.

##### Raises:


*  <b>`RuntimeError`</b>: If the model cannot be initialized or recovered.


- - -

#### `tf.train.SessionManager.recover_session(master, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None)` {#SessionManager.recover_session}

Creates a `Session`, recovering if possible.

Creates a new session on 'master'.  If the session is not initialized
and can be recovered from a checkpoint, recover it.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`saver`</b>: A `Saver` object used to restore a model.
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files.
*  <b>`wait_for_checkpoint`</b>: Whether to wait for checkpoint to become available.
*  <b>`max_wait_secs`</b>: Maximum time to wait for checkpoints to become available.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.

##### Returns:

  A pair (sess, initialized) where 'initialized' is `True` if
  the session could be recovered and initialized, `False` otherwise.


- - -

#### `tf.train.SessionManager.wait_for_session(master, config=None, max_wait_secs=inf)` {#SessionManager.wait_for_session}

Creates a new `Session` and waits for model to be ready.

Creates a new `Session` on 'master'.  Waits for the model to be
initialized or recovered from a checkpoint.  It's expected that
another thread or process will make the model ready, and that this
is intended to be used by threads/processes that participate in a
distributed training configuration where a different thread/process
is responsible for initializing or recovering the model being trained.

NB: The amount of time this method waits for the session is bounded
by max_wait_secs. By default, this function will wait indefinitely.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: Optional ConfigProto proto used to configure the session.
*  <b>`max_wait_secs`</b>: Maximum time to wait for the session to become available.

##### Returns:

  A `Session`. May be None if the operation exceeds the timeout
  specified by config.operation_timeout_in_ms.

##### Raises:

  tf.DeadlineExceededError: if the session is not available after
    max_wait_secs.



- - -

### `class tf.train.ClusterSpec` {#ClusterSpec}

Represents a cluster as a set of "tasks", organized into "jobs".

A `tf.train.ClusterSpec` represents the set of processes that
participate in a distributed TensorFlow computation. Every
[`tf.train.Server`](#Server) is constructed in a particular cluster.

To create a cluster with two jobs and five tasks, you specify the
mapping from job names to lists of network addresses (typically
hostname-port pairs).

```
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

Each job may also be specified as a sparse mapping from task indices
to network addresses. This enables a server to be configured without
needing to know the identity of (for example) all other worker
tasks:

```
cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

- - -

#### `tf.train.ClusterSpec.as_cluster_def()` {#ClusterSpec.as_cluster_def}

Returns a `tf.train.ClusterDef` protocol buffer based on this cluster.


- - -

#### `tf.train.ClusterSpec.as_dict()` {#ClusterSpec.as_dict}

Returns a dictionary from job names to their tasks.

For each job, if the task index space is dense, the corresponding
value will be a list of network addresses; otherwise it will be a
dictionary mapping (sparse) task indices to the corresponding
addresses.

##### Returns:

  A dictionary mapping job names to lists or dictionaries
  describing the tasks in those jobs.



#### Other Methods
- - -

#### `tf.train.ClusterSpec.__bool__()` {#ClusterSpec.__bool__}




- - -

#### `tf.train.ClusterSpec.__eq__(other)` {#ClusterSpec.__eq__}




- - -

#### `tf.train.ClusterSpec.__init__(cluster)` {#ClusterSpec.__init__}

Creates a `ClusterSpec`.

##### Args:


*  <b>`cluster`</b>: A dictionary mapping one or more job names to (i) a
    list of network addresses, or (ii) a dictionary mapping integer
    task indices to network addresses; or a `tf.train.ClusterDef`
    protocol buffer.

##### Raises:


*  <b>`TypeError`</b>: If `cluster` is not a dictionary mapping strings to lists
    of strings, and not a `tf.train.ClusterDef` protobuf.


- - -

#### `tf.train.ClusterSpec.__ne__(other)` {#ClusterSpec.__ne__}




- - -

#### `tf.train.ClusterSpec.__nonzero__()` {#ClusterSpec.__nonzero__}




- - -

#### `tf.train.ClusterSpec.job_tasks(job_name)` {#ClusterSpec.job_tasks}

Returns a mapping from task ID to address in the given job.

NOTE: For backwards compatibility, this method returns a list. If
the given job was defined with a sparse set of task indices, the
length of this list may not reflect the number of tasks defined in
this job. Use the [`num_tasks()`](#ClusterSpec.num_tasks) method
to find the number of tasks defined in a particular job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  A list of task addresses, where the index in the list
  corresponds to the task index of each task. The list may contain
  `None` if the job was defined with a sparse set of task indices.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster.


- - -

#### `tf.train.ClusterSpec.jobs` {#ClusterSpec.jobs}

Returns a list of job names in this cluster.

##### Returns:

  A list of strings, corresponding to the names of jobs in this cluster.


- - -

#### `tf.train.ClusterSpec.num_tasks(job_name)` {#ClusterSpec.num_tasks}

Returns the number of tasks defined in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  The number of tasks defined in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster.


- - -

#### `tf.train.ClusterSpec.task_address(job_name, task_index)` {#ClusterSpec.task_address}

Returns the address of the given task in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.
*  <b>`task_index`</b>: A non-negative integer.

##### Returns:

  The address of the given task in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster,
  or no task with index `task_index` is defined in that job.


- - -

#### `tf.train.ClusterSpec.task_indices(job_name)` {#ClusterSpec.task_indices}

Returns a list of valid task indices in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  A list of valid task indices in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster,
  or no task with index `task_index` is defined in that job.



- - -

### `tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None)` {#replica_device_setter}

Return a `device function` to use when building a Graph for replicas.

Device Functions are used in `with tf.device(device_function):` statement to
automatically assign devices to `Operation` objects as they are constructed,
Device constraints are added from the inner-most context first, working
outwards. The merging behavior adds constraints to fields that are yet unset
by a more inner context. Currently the fields are (job, task, cpu/gpu).

If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.

For example,

```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

##### Args:


*  <b>`ps_tasks`</b>: Number of tasks in the `ps` job.
*  <b>`ps_device`</b>: String.  Device of the `ps` job.  If empty no `ps` job is used.
    Defaults to `ps`.
*  <b>`worker_device`</b>: String.  Device of the `worker` job.  If empty no `worker`
    job is used.
*  <b>`merge_devices`</b>: `Boolean`. If `True`, merges or only sets a device if the
    device constraint is completely unset. merges device specification rather
    than overriding them.
*  <b>`cluster`</b>: `ClusterDef` proto or `ClusterSpec`.
*  <b>`ps_ops`</b>: List of `Operation` objects that need to be placed on `ps` devices.

##### Returns:

  A function to pass to `tf.device()`.

##### Raises:

  TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer.



## Summary Operations

The following ops output
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffers as serialized string tensors.

You can fetch the output of a summary op in a session, and pass it to
a [SummaryWriter](../../api_docs/python/train.md#SummaryWriter) to append it
to an event file.  Event files contain
[`Event`](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
protos that can contain `Summary` protos along with the timestamp and
step.  You can then use TensorBoard to visualize the contents of the
event files.  See [TensorBoard and
Summaries](../../how_tos/summaries_and_tensorboard/index.md) for more
details.

- - -

### `tf.scalar_summary(tags, values, collections=None, name=None)` {#scalar_summary}

Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated
summary has a summary value for each tag-value pair in `tags` and `values`.

##### Args:


*  <b>`tags`</b>: A `string` `Tensor`.  Tags for the summaries.
*  <b>`values`</b>: A real numeric Tensor.  Values for the summaries.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)` {#image_summary}

Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

##### Args:


*  <b>`tag`</b>: A scalar `Tensor` of type `string`. Used to build the `tag`
    of the summary values.
*  <b>`tensor`</b>: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
    width, channels]` where `channels` is 1, 3, or 4.
*  <b>`max_images`</b>: Max number of batch elements to generate images for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)` {#audio_summary}

Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
`sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

##### Args:


*  <b>`tag`</b>: A scalar `Tensor` of type `string`. Used to build the `tag`
    of the summary values.
*  <b>`tensor`</b>: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
    or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
*  <b>`sample_rate`</b>: The sample rate of the signal in hertz.
*  <b>`max_outputs`</b>: Max number of batch elements to generate audio for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.histogram_summary(tag, values, collections=None, name=None)` {#histogram_summary}

Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

##### Args:


*  <b>`tag`</b>: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
*  <b>`values`</b>: A real numeric `Tensor`. Any shape. Values to use to
    build the histogram.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.


- - -

### `tf.nn.zero_fraction(value, name=None)` {#zero_fraction}

Returns the fraction of zeros in `value`.

If `value` is empty, the result is `nan`.

This is useful in summaries to measure and report sparsity.  For example,

    z = tf.Relu(...)
    summ = tf.scalar_summary('sparsity', tf.nn.zero_fraction(z))

##### Args:


*  <b>`value`</b>: A tensor of numeric type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The fraction of zeros in `value`, with type `float32`.



- - -

### `tf.merge_summary(inputs, collections=None, name=None)` {#merge_summary}

Merges summaries.

This op creates a
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffer that contains the union of all the values in the input
summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values
in the summaries to merge use the same tag.

##### Args:


*  <b>`inputs`</b>: A list of `string` `Tensor` objects containing serialized `Summary`
    protocol buffers.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer resulting from the merging.


- - -

### `tf.merge_all_summaries(key='summaries')` {#merge_all_summaries}

Merges all summaries collected in the default graph.

##### Args:


*  <b>`key`</b>: `GraphKey` used to collect the summaries.  Defaults to
    `GraphKeys.SUMMARIES`.

##### Returns:

  If no summaries were collected, returns None.  Otherwise returns a scalar
  `Tensor` of type `string` containing the serialized `Summary` protocol
  buffer resulting from the merging.



## Adding Summaries to Event Files

See [Summaries and
TensorBoard](../../how_tos/summaries_and_tensorboard/index.md) for an
overview of summaries, event files, and visualization in TensorBoard.

- - -

### `class tf.train.SummaryWriter` {#SummaryWriter}

Writes `Summary` protocol buffers to event files.

The `SummaryWriter` class provides a mechanism to create an event file in a
given directory and add summaries and events to it. The class updates the
file contents asynchronously. This allows a training program to call methods
to add data to the file directly from the training loop, without slowing down
training.

- - -

#### `tf.train.SummaryWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)` {#SummaryWriter.__init__}

Creates a `SummaryWriter` and an event file.

On construction the summary writer creates a new event file in `logdir`.
This event file will contain `Event` protocol buffers constructed when you
call one of the following functions: `add_summary()`, `add_session_log()`,
`add_event()`, or `add_graph()`.

If you pass a `Graph` to the constructor it is added to
the event file. (This is equivalent to calling `add_graph()` later).

TensorBoard will pick the graph from the file and display it graphically so
you can interactively explore the graph you built. You will usually pass
the graph from the session in which you launched it:

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
```

The other arguments to the constructor control the asynchronous writes to
the event file:

*  `flush_secs`: How often, in seconds, to flush the added summaries
   and events to disk.
*  `max_queue`: Maximum number of summaries or events pending to be
   written to disk before one of the 'add' calls block.

##### Args:


*  <b>`logdir`</b>: A string. Directory where event file will be written.
*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`max_queue`</b>: Integer. Size of the queue for pending events and summaries.
*  <b>`flush_secs`</b>: Number. How often, in seconds, to flush the
    pending events and summaries to disk.
*  <b>`graph_def`</b>: DEPRECATED: Use the `graph` argument instead.



- - -

#### `tf.train.SummaryWriter.add_summary(summary, global_step=None)` {#SummaryWriter.add_summary}

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer
and adds it to the event file.

You can pass the result of evaluating any summary op, using
[`Session.run()`](client.md#Session.run) or
[`Tensor.eval()`](framework.md#Tensor.eval), to this
function. Alternatively, you can pass a `tf.Summary` protocol
buffer that you populate with your own data. The latter is
commonly done to report evaluation results in event files.

##### Args:


*  <b>`summary`</b>: A `Summary` protocol buffer, optionally serialized as a string.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.train.SummaryWriter.add_session_log(session_log, global_step=None)` {#SummaryWriter.add_session_log}

Adds a `SessionLog` protocol buffer to the event file.

This method wraps the provided session in an `Event` procotol buffer
and adds it to the event file.

##### Args:


*  <b>`session_log`</b>: A `SessionLog` protocol buffer.
*  <b>`global_step`</b>: Number. Optional global step value to record with the
    summary.


- - -

#### `tf.train.SummaryWriter.add_event(event)` {#SummaryWriter.add_event}

Adds an event to the event file.

##### Args:


*  <b>`event`</b>: An `Event` protocol buffer.


- - -

#### `tf.train.SummaryWriter.add_graph(graph, global_step=None, graph_def=None)` {#SummaryWriter.add_graph}

Adds a `Graph` to the event file.

The graph described by the protocol buffer will be displayed by
TensorBoard. Most users pass a graph in the constructor instead.

##### Args:


*  <b>`graph`</b>: A `Graph` object, such as `sess.graph`.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.
*  <b>`graph_def`</b>: DEPRECATED. Use the `graph` parameter instead.

##### Raises:


*  <b>`ValueError`</b>: If both graph and graph_def are passed to the method.


- - -

#### `tf.train.SummaryWriter.add_run_metadata(run_metadata, tag, global_step=None)` {#SummaryWriter.add_run_metadata}

Adds a metadata information for a single session.run() call.

##### Args:


*  <b>`run_metadata`</b>: A `RunMetadata` protobuf object.
*  <b>`tag`</b>: The tag name for this metadata.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    StepStats.

##### Raises:


*  <b>`ValueError`</b>: If the provided tag was already used for this type of event.


- - -

#### `tf.train.SummaryWriter.get_logdir()` {#SummaryWriter.get_logdir}

Returns the directory where event file will be written.



- - -

#### `tf.train.SummaryWriter.flush()` {#SummaryWriter.flush}

Flushes the event file to disk.

Call this method to make sure that all pending events have been written to
disk.


- - -

#### `tf.train.SummaryWriter.close()` {#SummaryWriter.close}

Flushes the event file to disk and close the file.

Call this method when you do not need the summary writer anymore.



#### Other Methods
- - -

#### `tf.train.SummaryWriter.reopen()` {#SummaryWriter.reopen}

Reopens the summary writer.

Can be called after `close()` to add more events in the same directory.
The events will go into a new events file.

Does nothing if the summary writer was not closed.



- - -

### `tf.train.summary_iterator(path)` {#summary_iterator}

An iterator for reading `Event` protocol buffers from an event file.

You can use this function to read events written to an event file. It returns
a Python iterator that yields `Event` protocol buffers.

Example: Print the contents of an events file.

```python
for e in tf.train.summary_iterator(path to events file):
    print(e)
```

Example: Print selected summary values.

```python
# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in tf.train.summary_iterator(path to events file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
```

See the protocol buffer definitions of
[Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
and
[Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
for more information about their attributes.

##### Args:


*  <b>`path`</b>: The path to an event file created by a `SummaryWriter`.

##### Yields:

  `Event` protocol buffers.



## Training utilities

- - -

### `tf.train.global_step(sess, global_step_tensor)` {#global_step}

Small helper to get the global step.

```python
# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
sess.run(global_step_tensor.initializer)
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

global_step: 10
```

##### Args:


*  <b>`sess`</b>: A TensorFlow `Session` object.
*  <b>`global_step_tensor`</b>: `Tensor` or the `name` of the operation that contains
    the global step.

##### Returns:

  The global step value.


- - -

### `tf.train.write_graph(graph_def, logdir, name, as_text=True)` {#write_graph}

Writes a graph proto to a file.

The graph is written as a binary proto unless `as_text` is `True`.

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` protocol buffer.
*  <b>`logdir`</b>: Directory where to write the graph. This can refer to remote
    filesystems, such as Google Cloud Storage (GCS).
*  <b>`name`</b>: Filename for the graph.
*  <b>`as_text`</b>: If `True`, writes the graph as an ASCII proto.



## Other Functions and Classes
- - -

### `class tf.train.LooperThread` {#LooperThread}

A thread that runs code repeatedly, optionally on a timer.

This thread class is intended to be used with a `Coordinator`.  It repeatedly
runs code specified either as `target` and `args` or by the `run_loop()`
method.

Before each run the thread checks if the coordinator has requested stop.  In
that case the looper thread terminates immediately.

If the code being run raises an exception, that exception is reported to the
coordinator and the thread terminates.  The coordinator will then request all
the other threads it coordinates to stop.

You typically pass looper threads to the supervisor `Join()` method.
- - -

#### `tf.train.LooperThread.__init__(coord, timer_interval_secs, target=None, args=None, kwargs=None)` {#LooperThread.__init__}

Create a LooperThread.

##### Args:


*  <b>`coord`</b>: A Coordinator.
*  <b>`timer_interval_secs`</b>: Time boundaries at which to call Run(), or None
    if it should be called back to back.
*  <b>`target`</b>: Optional callable object that will be executed in the thread.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.train.LooperThread.__repr__()` {#LooperThread.__repr__}




- - -

#### `tf.train.LooperThread.daemon` {#LooperThread.daemon}

A boolean value indicating whether this thread is a daemon thread (True) or not (False).

This must be set before start() is called, otherwise RuntimeError is
raised. Its initial value is inherited from the creating thread; the
main thread is not a daemon thread and therefore all threads created in
the main thread default to daemon = False.

The entire Python program exits when no alive non-daemon threads are
left.


- - -

#### `tf.train.LooperThread.getName()` {#LooperThread.getName}




- - -

#### `tf.train.LooperThread.ident` {#LooperThread.ident}

Thread identifier of this thread or None if it has not been started.

This is a nonzero integer. See the thread.get_ident() function. Thread
identifiers may be recycled when a thread exits and another thread is
created. The identifier is available even after the thread has exited.


- - -

#### `tf.train.LooperThread.isAlive()` {#LooperThread.isAlive}

Return whether the thread is alive.

This method returns True just before the run() method starts until just
after the run() method terminates. The module function enumerate()
returns a list of all alive threads.


- - -

#### `tf.train.LooperThread.isDaemon()` {#LooperThread.isDaemon}




- - -

#### `tf.train.LooperThread.is_alive()` {#LooperThread.is_alive}

Return whether the thread is alive.

This method returns True just before the run() method starts until just
after the run() method terminates. The module function enumerate()
returns a list of all alive threads.


- - -

#### `tf.train.LooperThread.join(timeout=None)` {#LooperThread.join}

Wait until the thread terminates.

This blocks the calling thread until the thread whose join() method is
called terminates -- either normally or through an unhandled exception
or until the optional timeout occurs.

When the timeout argument is present and not None, it should be a
floating point number specifying a timeout for the operation in seconds
(or fractions thereof). As join() always returns None, you must call
isAlive() after join() to decide whether a timeout happened -- if the
thread is still alive, the join() call timed out.

When the timeout argument is not present or None, the operation will
block until the thread terminates.

A thread can be join()ed many times.

join() raises a RuntimeError if an attempt is made to join the current
thread as that would cause a deadlock. It is also an error to join() a
thread before it has been started and attempts to do so raises the same
exception.


- - -

#### `tf.train.LooperThread.loop(coord, timer_interval_secs, target, args=None, kwargs=None)` {#LooperThread.loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(args)`
repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
seconds.  The thread terminates when a stop of the coordinator is
requested.

##### Args:


*  <b>`coord`</b>: A Coordinator.
*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.


- - -

#### `tf.train.LooperThread.name` {#LooperThread.name}

A string used for identification purposes only.

It has no semantics. Multiple threads may be given the same name. The
initial name is set by the constructor.


- - -

#### `tf.train.LooperThread.run()` {#LooperThread.run}




- - -

#### `tf.train.LooperThread.run_loop()` {#LooperThread.run_loop}

Called at 'timer_interval_secs' boundaries.


- - -

#### `tf.train.LooperThread.setDaemon(daemonic)` {#LooperThread.setDaemon}




- - -

#### `tf.train.LooperThread.setName(name)` {#LooperThread.setName}




- - -

#### `tf.train.LooperThread.start()` {#LooperThread.start}

Start the thread's activity.

It must be called at most once per thread object. It arranges for the
object's run() method to be invoked in a separate thread of control.

This method will raise a RuntimeError if called more than once on the
same thread object.


- - -

#### `tf.train.LooperThread.start_loop()` {#LooperThread.start_loop}

Called when the thread starts.


- - -

#### `tf.train.LooperThread.stop_loop()` {#LooperThread.stop_loop}

Called when the thread stops.



- - -

### `tf.train.do_quantize_training_on_graphdef(input_graph, num_bits)` {#do_quantize_training_on_graphdef}




- - -

### `tf.train.generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None)` {#generate_checkpoint_state_proto}

Generates a checkpoint state proto.

##### Args:


*  <b>`save_dir`</b>: Directory where the model was saved.
*  <b>`model_checkpoint_path`</b>: The checkpoint file.
*  <b>`all_model_checkpoint_paths`</b>: List of strings.  Paths to all not-yet-deleted
    checkpoints, sorted from oldest to newest.  If this is a non-empty list,
    the last element must be equal to model_checkpoint_path.  These paths
    are also saved in the CheckpointState proto.

##### Returns:

  CheckpointState proto with model_checkpoint_path and
  all_model_checkpoint_paths updated to either absolute paths or
  relative paths to the current save_dir.


