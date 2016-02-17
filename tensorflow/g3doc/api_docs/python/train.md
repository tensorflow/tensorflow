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
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1])) for gv in grads_and_vars]

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

#### `tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None)` {#Optimizer.minimize}

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

##### Returns:

  An Operation that updates the variables in `var_list`.  If `global_step`
  was not `None`, that operation also increments `global_step`.

##### Raises:


*  <b>`ValueError`</b>: If some of the variables are not `Variable` objects.


- - -

#### `tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False)` {#Optimizer.compute_gradients}

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

Both `minimize()` and `compute_gradients()` accept a `gate_gradient` argument
that controls the degree of parallelism during the application of the
gradients.

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

### `class tf.train.MomentumOptimizer` {#MomentumOptimizer}

Optimizer that implements the Momentum algorithm.

- - -

#### `tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum')` {#MomentumOptimizer.__init__}

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

- - -

#### `tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')` {#FtrlOptimizer.__init__}

Construct a new FTRL optimizer.

The Ftrl-proximal algorithm, abbreviated for Follow-the-regularized-leader,
is described in the paper [Ad Click Prediction: a View from the Trenches](
https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf).

It can give a good performance vs. sparsity tradeoff.

Ftrl-proximal uses its own global base learning rate and can behave like
Adagrad with `learning_rate_power=-0.5`, or like gradient descent with
`learning_rate_power=0.0`.

The effective learning rate is adjusted per parameter, relative to this
base learning rate as:

```
effective_learning_rate_i = (learning_rate /
    pow(k + summed_squared_gradients_for_i, learning_rate_power));
```

where k is the small constant `initial_accumulator_value`.

Note that the real regularization coefficient of `|w|^2` for objective
function is `1 / lambda_2` if specifying `l2 = lambda_2` as argument when
using this function.

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

Constructs symbolic partial derivatives of `ys` w.r.t. x in `xs`.

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

### `tf.clip_by_norm(t, clip_norm, name=None)` {#clip_by_norm}

Clips tensor values to a maximum L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
normalizes `t` so that its L2-norm is less than or equal to `clip_norm`.
Specifically, if the L2-norm is already less than or equal to `clip_norm`,
then `t` is not modified. If the L2-norm is greater than `clip_norm`, then
this operation returns a tensor of the same type and shape as `t` with its
values set to:

`t * clip_norm / l2norm(t)`

In this case, the L2-norm of the output tensor is `clip_norm`.

This operation is typically used to clip gradients before applying them with
an optimizer.

##### Args:


*  <b>`t`</b>: A `Tensor`.
*  <b>`clip_norm`</b>: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
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

If the argument `staircase` is `True`, then `global_step /decay_steps` is an
integer division and the decayed learning rate follows a staircase function.

Example: decay every 100000 steps with a base of 0.96:

```python
...
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
optimizer = tf.GradientDescentOptimizer(learning_rate)
# Passing global_step to minimize() will increment it at each step.
optimizer.minimize(...my loss..., global_step=global_step)
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
*  <b>`staircase`</b>: Boolean.  It `True` decay the learning rate at discrete intervals.
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to 'ExponentialDecay'

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

The `Apply()` method has to be called to create shadow variables and add
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
    `Apply()`.


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
    and Tensors must be of types float32 or float64.

##### Returns:

  An Operation that updates the moving averages.

##### Raises:


*  <b>`TypeError`</b>: If the arguments are not all float32 or float64.
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

#### `tf.train.ExponentialMovingAverage.variables_to_restore()` {#ExponentialMovingAverage.variables_to_restore}

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

#### `tf.train.Coordinator.__init__()` {#Coordinator.__init__}

Create a new Coordinator.


- - -

#### `tf.train.Coordinator.clear_stop()` {#Coordinator.clear_stop}

Clears the stop flag.

After this is called, calls to `should_stop()` will return `False`.


- - -

#### `tf.train.Coordinator.join(threads, stop_grace_period_secs=120)` {#Coordinator.join}

Wait for threads to terminate.

Blocks until all `threads` have terminated or `request_stop()` is called.

After the threads stop, if an `exc_info` was passed to `request_stop`, that
exception is re-raised.

Grace period handling: When `request_stop()` is called, threads are given
'stop_grace_period_secs' seconds to terminate.  If any of them is still
alive after that period expires, a `RuntimeError` is raised.  Note that if
an `exc_info` was passed to `request_stop()` then it is raised instead of
that `RuntimeError`.

##### Args:


*  <b>`threads`</b>: List of `threading.Threads`. The started threads to join.
*  <b>`stop_grace_period_secs`</b>: Number of seconds given to threads to stop after
    `request_stop()` has been called.

##### Raises:


*  <b>`RuntimeError`</b>: If any thread is still alive after `request_stop()`
    is called and the grace period expires.


- - -

#### `tf.train.Coordinator.request_stop(ex=None)` {#Coordinator.request_stop}

Request that the threads stop.

After this is called, calls to `should_stop()` will return `True`.

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

#### `tf.train.QueueRunner.__init__(queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_runner_def=None)` {#QueueRunner.__init__}

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




- - -

#### `tf.train.QueueRunner.name` {#QueueRunner.name}

The string name of the underlying Queue.


- - -

#### `tf.train.QueueRunner.queue` {#QueueRunner.queue}




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

### `tf.histogram_summary(tag, values, collections=None, name=None)` {#histogram_summary}

Outputs a `Summary` protocol buffer with a histogram.

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `OutOfRange` error if any value is not finite.

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
    summ = tf.scalar_summary('sparsity', tf.zero_fraction(z))

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
  `Tensor` of type`string` containing the serialized `Summary` protocol
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

#### `tf.train.SummaryWriter.__init__(logdir, graph_def=None, max_queue=10, flush_secs=120)` {#SummaryWriter.__init__}

Creates a `SummaryWriter` and an event file.

On construction the summary writer creates a new event file in `logdir`.
This event file will contain `Event` protocol buffers constructed when you
call one of the following functions: `add_summary()`, `add_session_log()`,
`add_event()`, or `add_graph()`.

If you pass a `graph_def` protocol buffer to the constructor it is added to
the event file. (This is equivalent to calling `add_graph()` later).

TensorBoard will pick the graph from the file and display it graphically so
you can interactively explore the graph you built. You will usually pass
the graph from the session in which you launched it:

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph_def' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph_def)
```

The other arguments to the constructor control the asynchronous writes to
the event file:

*  `flush_secs`: How often, in seconds, to flush the added summaries
   and events to disk.
*  `max_queue`: Maximum number of summaries or events pending to be
   written to disk before one of the 'add' calls block.

##### Args:


*  <b>`logdir`</b>: A string. Directory where event file will be written.
*  <b>`graph_def`</b>: A `GraphDef` protocol buffer.
*  <b>`max_queue`</b>: Integer. Size of the queue for pending events and summaries.
*  <b>`flush_secs`</b>: Number. How often, in seconds, to flush the
    pending events and summaries to disk.



- - -

#### `tf.train.SummaryWriter.add_summary(summary, global_step=None)` {#SummaryWriter.add_summary}

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer
and adds it to the event file.

You can pass the result of evaluating any summary op, using
[`Session.run()`](client.md#Session.run] or
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

#### `tf.train.SummaryWriter.add_graph(graph_def, global_step=None)` {#SummaryWriter.add_graph}

Adds a `GraphDef` protocol buffer to the event file.

The graph described by the protocol buffer will be displayed by
TensorBoard. Most users pass a graph in the constructor instead.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` protocol buffer.
*  <b>`global_step`</b>: Number. Optional global step counter to record with the
    graph.



- - -

#### `tf.train.SummaryWriter.flush()` {#SummaryWriter.flush}

Flushes the event file to disk.

Call this method to make sure that all pending events have been written to
disk.


- - -

#### `tf.train.SummaryWriter.close()` {#SummaryWriter.close}

Flushes the event file to disk and close the file.

Call this method when you do not need the summary writer anymore.



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


*  <b>`sess`</b>: A brain `Session` object.
*  <b>`global_step_tensor`</b>: `Tensor` or the `name` of the operation that contains
    the global step.

##### Returns:

  The global step value.


- - -

### `tf.train.write_graph(graph_def, logdir, name, as_text=True)` {#write_graph}

Writes a graph proto on disk.

The graph is written as a binary proto unless `as_text` is `True`.

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` protocol buffer.
*  <b>`logdir`</b>: Directory where to write the graph.
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

#### `tf.train.LooperThread.__init__(coord, timer_interval_secs, target=None, args=None)` {#LooperThread.__init__}

Create a LooperThread.

##### Args:


*  <b>`coord`</b>: A Coordinator.
*  <b>`timer_interval_secs`</b>: Time boundaries at which to call Run(), or None
    if it should be called back to back.
*  <b>`target`</b>: Optional callable object that will be executed in the thread.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


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

#### `tf.train.LooperThread.loop(coord, timer_interval_secs, target, args=None)` {#LooperThread.loop}

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

### `tf.train.export_meta_graph(filename=None, meta_info_def=None, graph_def=None, saver_def=None, collection_list=None, as_text=False)` {#export_meta_graph}

Returns `MetaGraphDef` proto. Optionally writes it to filename.

This function exports the graph, saver, and collection objects into
`MetaGraphDef` protocol buffer with the intension of it being imported
at a later time or location to restart training, run inference, or be
a subgraph.

##### Args:


*  <b>`filename`</b>: Optional filename including the path for writing the
    generated `MetaGraphDef` protocol buffer.
*  <b>`meta_info_def`</b>: `MetaInfoDef` protocol buffer.
*  <b>`graph_def`</b>: `GraphDef` protocol buffer.
*  <b>`saver_def`</b>: `SaverDef` protocol buffer.
*  <b>`collection_list`</b>: List of string keys to collect.
*  <b>`as_text`</b>: If `True`, writes the `MetaGraphDef` as an ASCII proto.

##### Returns:

  A `MetaGraphDef` proto.


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


- - -

### `tf.train.import_meta_graph(meta_graph_or_file)` {#import_meta_graph}

Recreates a Graph saved in a `MetaGraphDef` proto.

This function reads from a file containing a `MetaGraphDef` proto,
adds all the nodes from the graph_def proto to the current graph,
recreates all the collections, and returns a saver from saver_def.

In combination with `export_meta_graph()`, this function can be used to

* Serialize a graph along with other Python objects such as `QueueRunner`,
  `Variable` into a `MetaGraphDef`.

* Restart training from a saved graph and checkpoints.

* Run inference from a saved graph and checkpoints.

##### Args:


*  <b>`meta_graph_or_file`</b>: `MetaGraphDef` protocol buffer or filename (including
    the path) containing a `MetaGraphDef`.

##### Returns:

  A saver constructed rom `saver_def` in `MetaGraphDef`.


