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
*  <b>`use_nesterov`</b>: If `True` use Nesterov Momentum.
    See [Sutskever et. al., 2013](
*  <b>`http`</b>: //jmlr.org/proceedings/papers/v28/sutskever13.pdf)



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

### `class tf.train.ProximalGradientDescentOptimizer` {#ProximalGradientDescentOptimizer}

Optimizer that implements the proximal gradient descent algorithm.

See this [paper](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).

- - -

#### `tf.train.ProximalGradientDescentOptimizer.__init__(learning_rate, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalGradientDescent')` {#ProximalGradientDescentOptimizer.__init__}

Construct a new proximal gradient descent optimizer.

##### Args:


*  <b>`learning_rate`</b>: A Tensor or a floating point value.  The learning
    rate to use.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If True use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients. Defaults to "GradientDescent".



- - -

### `class tf.train.ProximalAdagradOptimizer` {#ProximalAdagradOptimizer}

Optimizer that implements the Proximal Adagrad algorithm.

See this [paper](http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).

- - -

#### `tf.train.ProximalAdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='ProximalAdagrad')` {#ProximalAdagradOptimizer.__init__}

Construct a new ProximalAdagrad optimizer.

##### Args:


*  <b>`learning_rate`</b>: A `Tensor` or a floating point value.  The learning rate.
*  <b>`initial_accumulator_value`</b>: A floating point value.
    Starting value for the accumulators, must be positive.
*  <b>`l1_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`l2_regularization_strength`</b>: A float value, must be greater than or
    equal to zero.
*  <b>`use_locking`</b>: If `True` use locks for update operations.
*  <b>`name`</b>: Optional name prefix for the operations created when applying
    gradients.  Defaults to "Adagrad".

##### Raises:


*  <b>`ValueError`</b>: If the `initial_accumulator_value` is invalid.



- - -

### `class tf.train.RMSPropOptimizer` {#RMSPropOptimizer}

Optimizer that implements the RMSProp algorithm.

See the [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

- - -

#### `tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')` {#RMSPropOptimizer.__init__}

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
*  <b>`centered`</b>: If True, gradients are normalized by the estimated variance of
    the gradient; if False, by the uncentered second moment. Setting this to
    True may help with training, but is slightly more expensive in terms of
    computation and memory. Defaults to False.
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



- - -

### `tf.hessians(ys, xs, name='hessians', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)` {#hessians}

Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.

`hessians()` adds ops to the graph to output the Hessian matrix of `ys`
with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
where each tensor is the Hessian of `sum(ys)`. This function currently
only supports evaluating the Hessian with respect to (a list of) one-
dimensional tensors.

The Hessian is a matrix of second-order partial derivatives of a scalar
tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).

##### Args:


*  <b>`ys`</b>: A `Tensor` or list of tensors to be differentiated.
*  <b>`xs`</b>: A `Tensor` or list of tensors to be used for differentiation.
*  <b>`name`</b>: Optional name to use for grouping all the gradient ops together.
    defaults to 'hessians'.
*  <b>`colocate_gradients_with_ops`</b>: See `gradients()` documentation for details.
*  <b>`gate_gradients`</b>: See `gradients()` documentation for details.
*  <b>`aggregation_method`</b>: See `gradients()` documentation for details.

##### Returns:

  A list of Hessian matrices of `sum(y)` for each `x` in `xs`.

##### Raises:


*  <b>`LookupError`</b>: if one of the operations between `xs` and `ys` does not
    have a registered gradient function.
*  <b>`ValueError`</b>: if the arguments are invalid or not supported. Currently,
    this function only supports one-dimensional `x` in `xs`.




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
*  <b>`staircase`</b>: Boolean.  If `True` decay the learning rate at discrete intervals
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to
    'ExponentialDecay'.

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.

##### Raises:


*  <b>`ValueError`</b>: if `global_step` is not supplied.


- - -

### `tf.train.inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)` {#inverse_time_decay}

Applies inverse time decay to the initial learning rate.

When training a model, it is often recommended to lower the learning rate as
the training progresses.  This function applies an inverse decay function
to a provided initial learning rate.  It requires an `global_step` value to
compute the decayed learning rate.  You can just pass a TensorFlow variable
that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
decayed_learning_rate = learning_rate / (1 + decay_rate * t)
```

Example: decay 1/t with a rate of 0.5:

```python
...
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1
k = 0.5
learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k)

# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

##### Args:


*  <b>`learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate.
*  <b>`global_step`</b>: A Python number.
    Global step to use for the decay computation.  Must not be negative.
*  <b>`decay_steps`</b>: How often to apply decay.
*  <b>`decay_rate`</b>: A Python number.  The decay rate.
*  <b>`staircase`</b>: Whether to apply decay in a discrete staircase, as opposed to
    continuous, fashion.
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to
    'InverseTimeDecay'.

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.

##### Raises:


*  <b>`ValueError`</b>: if `global_step` is not supplied.


- - -

### `tf.train.natural_exp_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)` {#natural_exp_decay}

Applies natural exponential decay to the initial learning rate.

When training a model, it is often recommended to lower the learning rate as
the training progresses.  This function applies an exponential decay function
to a provided initial learning rate.  It requires an `global_step` value to
compute the decayed learning rate.  You can just pass a TensorFlow variable
that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
```

Example: decay exponentially with a base of 0.96:

```python
...
global_step = tf.Variable(0, trainable=False)
learning_rate = 0.1
k = 0.5
learning_rate = tf.train.exponential_time_decay(learning_rate, global_step, k)

# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

##### Args:


*  <b>`learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The initial learning rate.
*  <b>`global_step`</b>: A Python number.
    Global step to use for the decay computation.  Must not be negative.
*  <b>`decay_steps`</b>: How often to apply decay.
*  <b>`decay_rate`</b>: A Python number.  The decay rate.
*  <b>`staircase`</b>: Whether to apply decay in a discrete staircase, as opposed to
    continuous, fashion.
*  <b>`name`</b>: String.  Optional name of the operation.  Defaults to
    'ExponentialTimeDecay'.

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.

##### Raises:


*  <b>`ValueError`</b>: if `global_step` is not supplied.


- - -

### `tf.train.piecewise_constant(x, boundaries, values, name=None)` {#piecewise_constant}

Piecewise constant from boundaries and interval values.

Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5
  for steps 100001 to 110000, and 0.1 for any additional steps.

```python
global_step = tf.Variable(0, trainable=False)
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

# Later, whenever we perform an optimization step, we increment global_step.
```

##### Args:


*  <b>`x`</b>: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
    `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
*  <b>`boundaries`</b>: A list of `Tensor`s or `int`s or `float`s with strictly
    increasing entries, and with all elements having the same type as `x`.
*  <b>`values`</b>: A list of `Tensor`s or float`s or `int`s that specifies the values
    for the intervals defined by `boundaries`. It should have one more element
    than `boundaries`, and all elements should have the same type.
*  <b>`name`</b>: A string. Optional name of the operation. Defaults to
    'PiecewiseConstant'.

##### Returns:

  A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
  `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
  and values[-1] when `x > boundaries[-1]`.

##### Raises:


*  <b>`ValueError`</b>: if types of `x` and `buondaries` do not match, or types of all
      `values` do not match.


- - -

### `tf.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0, cycle=False, name=None)` {#polynomial_decay}

Applies a polynomial decay to the learning rate.

It is commonly observed that a monotonically decreasing learning rate, whose
degree of change is carefully chosen, results in a better performing model.
This function applies a polynomial decay function to a provided initial
`learning_rate` to reach an `end_learning_rate` in the given `decay_steps`.

It requires a `global_step` value to compute the decayed learning rate.  You
can just pass a TensorFlow variable that you increment at each training step.

The function returns the decayed learning rate.  It is computed as:

```python
global_step = min(global_step, decay_steps)
decayed_learning_rate = (learning_rate - end_learning_rate) *
                        (1 - global_step / decay_steps) ^ (power) +
                        end_learning_rate

```

If `cycle` is True then a multiple of `decay_steps` is used, the first one
that is bigger than `global_steps`.

```python
decay_steps = decay_steps * ceil(global_step / decay_steps)
decayed_learning_rate = (learning_rate - end_learning_rate) *
                        (1 - global_step / decay_steps) ^ (power) +
                        end_learning_rate

```

Example: decay from 0.1 to 0.01 in 10000 steps using sqrt (i.e. power=0.5):

```python
...
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.5)
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
*  <b>`end_learning_rate`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The minimal end learning rate.
*  <b>`power`</b>: A scalar `float32` or `float64` `Tensor` or a
    Python number.  The power of the polynomial. Defaults to sqrt, i.e. 0.5.
*  <b>`cycle`</b>: A boolean, whether or not it should cycle beyond decay_steps.
*  <b>`name`</b>: String.  Optional name of the operation. Defaults to
    'PolynomialDecay'.

##### Returns:

  A scalar `Tensor` of the same type as `learning_rate`.  The decayed
  learning rate.

##### Raises:


*  <b>`ValueError`</b>: if `global_step` is not supplied.



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

#### `tf.train.ExponentialMovingAverage.__init__(decay, num_updates=None, zero_debias=False, name='ExponentialMovingAverage')` {#ExponentialMovingAverage.__init__}

Creates a new ExponentialMovingAverage object.

The `apply()` method has to be called to create shadow variables and add
ops to maintain moving averages.

The optional `num_updates` parameter allows one to tweak the decay rate
dynamically. It is typical to pass the count of training steps, usually
kept in a variable that is incremented at each step, in which case the
decay rate is lower at the start of training.  This makes moving averages
move faster.  If passed, the actual decay rate used is:

  `min(decay, (1 + num_updates) / (10 + num_updates))`

##### Args:


*  <b>`decay`</b>: Float.  The decay to use.
*  <b>`num_updates`</b>: Optional count of number of updates applied to variables.
*  <b>`zero_debias`</b>: If `True`, zero debias moving-averages that are initialized
    with tensors.
*  <b>`name`</b>: String. Optional prefix name to use for the name of ops added in
    `apply()`.


- - -

#### `tf.train.ExponentialMovingAverage.apply(var_list=None)` {#ExponentialMovingAverage.apply}

Maintains moving averages of variables.

`var_list` must be a list of `Variable` or `Tensor` objects.  This method
creates shadow variables for all elements of `var_list`.  Shadow variables
for `Variable` objects are initialized to the variable's initial value.
They will be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
For `Tensor` objects, the shadow variables are initialized to 0 and zero
debiased (see docstring in `assign_moving_average` for more details).

shadow variables are created with `trainable=False` and added to the
`GraphKeys.ALL_VARIABLES` collection.  They will be returned by calls to
`tf.global_variables()`.

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
  is not maintained.


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

#### `tf.train.Coordinator.raise_requested_exception()` {#Coordinator.raise_requested_exception}

If an exception has been passed to `request_stop`, this raises it.


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

#### `tf.train.QueueRunner.__init__(queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None, queue_runner_def=None, import_scope=None)` {#QueueRunner.__init__}

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
*  <b>`import_scope`</b>: Optional `string`. Name scope to add. Only used when
    initializing from protocol buffer.

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

Create threads to run the enqueue ops for the given session.

This method requires a session in which the graph was launched.  It creates
a list of threads, optionally starting them.  There is one thread for each
op passed in `enqueue_ops`.

The `coord` argument is an optional coordinator that the threads will use
to terminate together and report exceptions.  If a coordinator is given,
this method starts an additional thread to close the queue when the
coordinator requests a stop.

If previously created threads for the given session are still running, no
new threads will be created.

##### Args:


*  <b>`sess`</b>: A `Session`.
*  <b>`coord`</b>: Optional `Coordinator` object for reporting errors and checking
    stop conditions.
*  <b>`daemon`</b>: Boolean.  If `True` make the threads daemon threads.
*  <b>`start`</b>: Boolean.  If `True` starts the threads.  If `False` the
    caller must call the `start()` method of the returned threads.

##### Returns:

  A list of threads.


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

#### `tf.train.QueueRunner.from_proto(queue_runner_def, import_scope=None)` {#QueueRunner.from_proto}

Returns a `QueueRunner` object created from `queue_runner_def`.


- - -

#### `tf.train.QueueRunner.name` {#QueueRunner.name}

The string name of the underlying Queue.


- - -

#### `tf.train.QueueRunner.queue` {#QueueRunner.queue}




- - -

#### `tf.train.QueueRunner.queue_closed_exception_types` {#QueueRunner.queue_closed_exception_types}




- - -

#### `tf.train.QueueRunner.to_proto(export_scope=None)` {#QueueRunner.to_proto}

Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

##### Args:


*  <b>`export_scope`</b>: Optional `string`. Name scope to remove.

##### Returns:

  A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in
  the specified name scope.



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
    `tf.report_uninitialized_variables(tf.global_variables())`. If `None`, the
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
    operation returned from summary.merge_all().  If `None`, summaries are
    not computed automatically.
*  <b>`saver`</b>: A Saver object.  Used by chief supervisors if a `logdir` was
    specified.  Defaults to the saved returned by Saver().
    If `None`, the model is not saved automatically.
*  <b>`global_step`</b>: An integer Tensor of size 1 that counts steps.  The value
    from 'global_step' is used in summaries and checkpoint filenames.
    Default to the op named 'global_step' in the graph if it exists, is of
    rank 1, size 1, and of type tf.int32 or tf.int64.  If `None` the global
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
model is considered ready if that operation returns an empty 1D string
tensor. If the operation returns a non empty 1D string tensor, the elements
are concatenated and used to indicate to the user why the model is not
ready.

The `ready_for_local_init_op` is an `Operation` used to check if the model
is ready to run local_init_op.  The model is considered ready if that
operation returns an empty 1D string tensor. If the operation returns a non
empty 1D string tensor, the elements are concatenated and used to indicate
to the user why the model is not ready.

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

#### `tf.train.SessionManager.prepare_session(master, init_op=None, saver=None, checkpoint_dir=None, checkpoint_filename_with_path=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None)` {#SessionManager.prepare_session}

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
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files. The latest checkpoint in the
    dir will be used to restore.
*  <b>`checkpoint_filename_with_path`</b>: Full file name path to the checkpoint file.
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

##### Raises:


*  <b>`ValueError`</b>: If both checkpoint_dir and checkpoint_filename_with_path are
    set.


- - -

#### `tf.train.SessionManager.recover_session(master, saver=None, checkpoint_dir=None, checkpoint_filename_with_path=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None)` {#SessionManager.recover_session}

Creates a `Session`, recovering if possible.

Creates a new session on 'master'.  If the session is not initialized
and can be recovered from a checkpoint, recover it.

##### Args:


*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`saver`</b>: A `Saver` object used to restore a model.
*  <b>`checkpoint_dir`</b>: Path to the checkpoint files. The latest checkpoint in the
    dir will be used to restore.
*  <b>`checkpoint_filename_with_path`</b>: Full file name path to the checkpoint file.
*  <b>`wait_for_checkpoint`</b>: Whether to wait for checkpoint to become available.
*  <b>`max_wait_secs`</b>: Maximum time to wait for checkpoints to become available.
*  <b>`config`</b>: Optional `ConfigProto` proto used to configure the session.

##### Returns:

  A pair (sess, initialized) where 'initialized' is `True` if
  the session could be recovered and initialized, `False` otherwise.

##### Raises:


*  <b>`ValueError`</b>: If both checkpoint_dir and checkpoint_filename_with_path are
    set.


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

```python
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

```python
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

### `tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None, ps_strategy=None)` {#replica_device_setter}

Return a `device function` to use when building a Graph for replicas.

Device Functions are used in `with tf.device(device_function):` statement to
automatically assign devices to `Operation` objects as they are constructed,
Device constraints are added from the inner-most context first, working
outwards. The merging behavior adds constraints to fields that are yet unset
by a more inner context. Currently the fields are (job, task, cpu/gpu).

If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.
Otherwise, the value of `ps_tasks` is derived from `cluster`.

By default, only Variable ops are placed on ps tasks, and the placement
strategy is round-robin over all ps tasks. A custom `ps_strategy` may be used
to do more intelligent placement, such as
`tf.contrib.training.GreedyLoadBalancingStrategy`.

For example,

```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

##### Args:


*  <b>`ps_tasks`</b>: Number of tasks in the `ps` job.  Ignored if `cluster` is
    provided.
*  <b>`ps_device`</b>: String.  Device of the `ps` job.  If empty no `ps` job is used.
    Defaults to `ps`.
*  <b>`worker_device`</b>: String.  Device of the `worker` job.  If empty no `worker`
    job is used.
*  <b>`merge_devices`</b>: `Boolean`. If `True`, merges or only sets a device if the
    device constraint is completely unset. merges device specification rather
    than overriding them.
*  <b>`cluster`</b>: `ClusterDef` proto or `ClusterSpec`.
*  <b>`ps_ops`</b>: List of strings representing `Operation` types that need to be
    placed on `ps` devices.  If `None`, defaults to `["Variable"]`.
*  <b>`ps_strategy`</b>: A callable invoked for every ps `Operation` (i.e. matched by
    `ps_ops`), that takes the `Operation` and returns the ps task index to
    use.  If `None`, defaults to a round-robin strategy across all `ps`
    devices.

##### Returns:

  A function to pass to `tf.device()`.

##### Raises:

  TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,
  or if `ps_strategy` is provided but not a callable.


- - -

### `tf.train.MonitoredTrainingSession(master='', is_chief=True, checkpoint_dir=None, scaffold=None, hooks=None, chief_only_hooks=None, save_checkpoint_secs=600, save_summaries_steps=100, config=None)` {#MonitoredTrainingSession}

Creates a `MonitoredSession` for training.

For a chief, this utility sets proper session initializer/restorer. It also
creates hooks related to checkpoint and summary saving. For workers, this
utility sets proper session creator which waits for the chief to
inialize/restore.


##### Args:


*  <b>`master`</b>: `String` the TensorFlow master to use.
*  <b>`is_chief`</b>: If `True`, it will take care of initialization and recovery the
    underlying TensorFlow session. If `False`, it will wait on a chief to
    initialize or recover the TensorFlow session.
*  <b>`checkpoint_dir`</b>: A string.  Optional path to a directory where to restore
    variables.
*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified, a default one is created. It's used to finalize the graph.
*  <b>`hooks`</b>: Optional list of `SessionRunHook` objects.
*  <b>`chief_only_hooks`</b>: list of `SessionRunHook` objects. Activate these hooks if
    `is_chief==True`, ignore otherwise.
*  <b>`save_checkpoint_secs`</b>: The frequency, in seconds, that a checkpoint is saved
    using a default checkpoint saver. If `save_checkpoint_secs` is set to
    `None`, then the default checkpoint saver isn't used.
*  <b>`save_summaries_steps`</b>: The frequency, in number of global steps, that the
    summaries are written to disk using a default summary saver. If
    `save_summaries_steps` is set to `None`, then the default summary saver
    isn't used.
*  <b>`config`</b>: an instance of `tf.ConfigProto` proto used to configure the session.
    It's the `config` argument of constructor of `tf.Session`.

##### Returns:

  A `MonitoredSession` object.


- - -

### `class tf.train.MonitoredSession` {#MonitoredSession}

Session-like object that handles initialization, recovery and hooks.

Example usage:

```python
saver_hook = CheckpointSaverHook(...)
summary_hook = SummaryHook(...)
with MonitoredSession(session_creator=ChiefSessionCreator(...),
                      hooks=[saver_hook, summary_hook]) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```

Initialization: At creation time the monitored session does following things
in given order:

* calls `hook.begin()` for each given hook
* finalizes the graph via `scaffold.finalize()`
* create session
* initializes the model via initialization ops provided by `Scaffold`
* restores variables if a checkpoint exists
* launches queue runners

Run: When `run()` is called, the monitored session does following things:

* calls `hook.before_run()`
* calls TensorFlow `session.run()` with merged fetches and feed_dict
* calls `hook.after_run()`
* returns result of `session.run()` asked by user
* if `AbortedError` occurs, it recovers or reinitializes the session before
  executing the run() call again


Exit: At the `close()`, the monitored session does following things in order:

* calls `hook.end()`
* closes the queue runners and the session
* suppresses `OutOfRange` error which indicates that all inputs have been
  processed if the monitored_session is used as a context

How to set `tf.Session` arguments:

* In most cases you can set session arguments as follows:

```python
MonitoredSession(
  session_creator=ChiefSessionCreator(master=..., config=...))
```

* In distributed setting for a non-chief worker, you can use following:

```python
MonitoredSession(
  session_creator=WorkerSessionCreator(master=..., config=...))
```

See `MonitoredTrainingSession` for an example usage based on chief or worker.

Args:
  session_creator: A factory object to create session. Typically a
    `ChiefSessionCreator` which is the default one.
  hooks: An iterable of `SessionRunHook' objects.

Returns:
  A MonitoredSession object.
- - -

#### `tf.train.MonitoredSession.__enter__()` {#MonitoredSession.__enter__}




- - -

#### `tf.train.MonitoredSession.__exit__(exception_type, exception_value, traceback)` {#MonitoredSession.__exit__}




- - -

#### `tf.train.MonitoredSession.__init__(session_creator=None, hooks=None)` {#MonitoredSession.__init__}




- - -

#### `tf.train.MonitoredSession.close()` {#MonitoredSession.close}




- - -

#### `tf.train.MonitoredSession.graph` {#MonitoredSession.graph}

The graph that was launched in this session.


- - -

#### `tf.train.MonitoredSession.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#MonitoredSession.run}

Run ops in the monitored session.

This method is completely compatible with the `tf.Session.run()` method.

##### Args:


*  <b>`fetches`</b>: Same as `tf.Session.run()`.
*  <b>`feed_dict`</b>: Same as `tf.Session.run()`.
*  <b>`options`</b>: Same as `tf.Session.run()`.
*  <b>`run_metadata`</b>: Same as `tf.Session.run()`.

##### Returns:

  Same as `tf.Session.run()`.


- - -

#### `tf.train.MonitoredSession.should_stop()` {#MonitoredSession.should_stop}





- - -

### `class tf.train.SingularMonitoredSession` {#SingularMonitoredSession}

Session-like object that handles initialization, restoring, and hooks.

Please note that this utility is not recommended for distributed settings.
For distributed settings, please use `tf.train.MonitoredSession`. The
differences between `MonitoredSession` and `SingularMonitoredSession` are:
* `MonitoredSession` handles `AbortedError` for distributed settings,
  but `SingularMonitoredSession` does not.
* `MonitoredSession` can be created in `chief` or `worker` modes.
  `SingularMonitoredSession` is always created as `chief`.
* You can access the raw `tf.Session` object used by
  `SingularMonitoredSession`, whereas in MonitoredSession the raw session is
  private. This can be used:
  - To `run` without hooks.
  - To save and restore.
* All other functionality is identical.

Example usage:
```python
saver_hook = CheckpointSaverHook(...)
summary_hook = SummaryHook(...)
with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
  while not sess.should_stop():
    sess.run(train_op)
```

Initialization: At creation time the hooked session does following things
in given order:

* calls `hook.begin()` for each given hook
* finalizes the graph via `scaffold.finalize()`
* create session
* initializes the model via initialization ops provided by `Scaffold`
* restores variables if a checkpoint exists
* launches queue runners

Run: When `run()` is called, the hooked session does following things:

* calls `hook.before_run()`
* calls TensorFlow `session.run()` with merged fetches and feed_dict
* calls `hook.after_run()`
* returns result of `session.run()` asked by user

Exit: At the `close()`, the hooked session does following things in order:

* calls `hook.end()`
* closes the queue runners and the session
* surpresses `OutOfRange` error which indicates that all inputs have been
  processed if the `SingularMonitoredSession` is used as a context.
- - -

#### `tf.train.SingularMonitoredSession.__enter__()` {#SingularMonitoredSession.__enter__}




- - -

#### `tf.train.SingularMonitoredSession.__exit__(exception_type, exception_value, traceback)` {#SingularMonitoredSession.__exit__}




- - -

#### `tf.train.SingularMonitoredSession.__init__(hooks=None, scaffold=None, master='', config=None, checkpoint_dir=None)` {#SingularMonitoredSession.__init__}

Creates a SingularMonitoredSession.

##### Args:


*  <b>`hooks`</b>: An iterable of `SessionRunHook' objects.
*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified a default one is created. It's used to finalize the graph.
*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.
*  <b>`checkpoint_dir`</b>: A string.  Optional path to a directory where to restore
    variables.


- - -

#### `tf.train.SingularMonitoredSession.close()` {#SingularMonitoredSession.close}




- - -

#### `tf.train.SingularMonitoredSession.graph` {#SingularMonitoredSession.graph}

The graph that was launched in this session.


- - -

#### `tf.train.SingularMonitoredSession.raw_session()` {#SingularMonitoredSession.raw_session}

Returns underlying `TensorFlow.Session` object.


- - -

#### `tf.train.SingularMonitoredSession.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#SingularMonitoredSession.run}

Run ops in the monitored session.

This method is completely compatible with the `tf.Session.run()` method.

##### Args:


*  <b>`fetches`</b>: Same as `tf.Session.run()`.
*  <b>`feed_dict`</b>: Same as `tf.Session.run()`.
*  <b>`options`</b>: Same as `tf.Session.run()`.
*  <b>`run_metadata`</b>: Same as `tf.Session.run()`.

##### Returns:

  Same as `tf.Session.run()`.


- - -

#### `tf.train.SingularMonitoredSession.should_stop()` {#SingularMonitoredSession.should_stop}





- - -

### `class tf.train.Scaffold` {#Scaffold}

Structure to create or gather pieces commonly needed to train a model.

When you build a model for training you usually need ops to initialize
variables, a `Saver` to checkpoint them, an op to collect summaries for
the visualizer, and so on.

Various libraries built on top of the core TensorFlow library take care of
creating some or all of these pieces and storing them in well known
collections in the graph.  The `Scaffold` class helps pick these pieces from
the graph collections, creating and adding them to the collections if needed.

If you call the scaffold constructor without any arguments, it will pick
pieces from the collections, creating default ones if needed when
`scaffold.finalize()` is called.  You can pass arguments to the constructor to
provide your own pieces.  Pieces that you pass to the constructor are not
added to the graph collections.

The following pieces are directly accessible as attributes of the `Scaffold`
object:

* `saver`: A `tf.Saver` object taking care of saving the variables.  Picked
  from and stored into the `SAVERS` collection in the graph by default.
* `init_op`: An op to run to initialize the variables.  Picked from and
  stored into the `INIT_OP` collection in the graph by default.
* `ready_op`: An op to verify that the variables are initialized.  Picked
  from and stored into the `READY_OP` collection in the graph by default.
* `ready_for_local_init_op`: An op to verify that global state has been
  initialized and it is alright to run `local_init_op`.  Picked from and
  stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
  default. This is needed when the initialization of local variables depends
  on the values of global variables.
* `local_init_op`: An op to initialize the local variables.  Picked
  from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
* `summary_op`: An op to run and merge the summaries in the graph.  Picked
  from and stored into the `SUMMARY_OP` collection in the graph by default.
* `global_step`: A tensor containing the global step counter.  Picked
  from and stored into the `GLOBAL_STEP` collection in the graph by default.

You can also pass the following additional pieces to the constructor:

* `init_feed_dict`: A sessionn feed dictionary that should be used when
   running the init op.
* `init_fn`: A callable to run run after the init op to perform additional
  initializations.  The callable will be called as
  `init_fn(scaffold, session)`.
- - -

#### `tf.train.Scaffold.__init__(init_op=None, init_feed_dict=None, init_fn=None, ready_op=None, ready_for_local_init_op=None, local_init_op=None, summary_op=None, saver=None)` {#Scaffold.__init__}

Create a scaffold.

##### Args:


*  <b>`init_op`</b>: Optional op for initializing variables.
*  <b>`init_feed_dict`</b>: Optional session feed dictionary to use when running the
    init_op.
*  <b>`init_fn`</b>: Optional function to use to initialize the model after running
    the init_op.  Will be called as `init_fn(scaffold, session)`.
*  <b>`ready_op`</b>: Optional op to verify that the variables are initialized.  Must
    return an empty 1D string tensor when the variables are initialized, or
    a non-empty 1D string tensor listing the names of the non-initialized
    variables.
*  <b>`ready_for_local_init_op`</b>: Optional op to verify that the global variables
    are initialized and `local_init_op` can be run. Must return an empty
    1D string tensor when the global variables are initialized, or a
    non-empty 1D string tensor listing the names of the non-initialized
    global variables.
*  <b>`local_init_op`</b>: Optional op to initialize local variables.
*  <b>`summary_op`</b>: Optional op to gather all summaries.  Must return a scalar
    string tensor containing a serialized `Summary` proto.
*  <b>`saver`</b>: Optional `tf.Saver` object to use to save and restore variables.


- - -

#### `tf.train.Scaffold.finalize()` {#Scaffold.finalize}

Creates operations if needed and finalizes the graph.


- - -

#### `tf.train.Scaffold.get_or_default(arg_name, collection_key, default_constructor)` {#Scaffold.get_or_default}

Get from cache or create a default operation.


- - -

#### `tf.train.Scaffold.init_feed_dict` {#Scaffold.init_feed_dict}




- - -

#### `tf.train.Scaffold.init_fn` {#Scaffold.init_fn}




- - -

#### `tf.train.Scaffold.init_op` {#Scaffold.init_op}




- - -

#### `tf.train.Scaffold.local_init_op` {#Scaffold.local_init_op}




- - -

#### `tf.train.Scaffold.ready_for_local_init_op` {#Scaffold.ready_for_local_init_op}




- - -

#### `tf.train.Scaffold.ready_op` {#Scaffold.ready_op}




- - -

#### `tf.train.Scaffold.saver` {#Scaffold.saver}




- - -

#### `tf.train.Scaffold.summary_op` {#Scaffold.summary_op}





- - -

### `class tf.train.SessionCreator` {#SessionCreator}

A factory for tf.Session.
- - -

#### `tf.train.SessionCreator.create_session()` {#SessionCreator.create_session}





- - -

### `class tf.train.ChiefSessionCreator` {#ChiefSessionCreator}

Creates a tf.Session  for a chief.
- - -

#### `tf.train.ChiefSessionCreator.__init__(scaffold=None, master='', config=None, checkpoint_dir=None, checkpoint_filename_with_path=None)` {#ChiefSessionCreator.__init__}

Initializes a chief session creator.

##### Args:


*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified a default one is created. It's used to finalize the graph.
*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.
*  <b>`checkpoint_dir`</b>: A string.  Optional path to a directory where to restore
    variables.
*  <b>`checkpoint_filename_with_path`</b>: Full file name path to the checkpoint file.


- - -

#### `tf.train.ChiefSessionCreator.create_session()` {#ChiefSessionCreator.create_session}





- - -

### `class tf.train.WorkerSessionCreator` {#WorkerSessionCreator}

Creates a tf.Session for a worker.
- - -

#### `tf.train.WorkerSessionCreator.__init__(scaffold=None, master='', config=None)` {#WorkerSessionCreator.__init__}

Initializes a worker session creator.

##### Args:


*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified a default one is created. It's used to finalize the graph.
*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.


- - -

#### `tf.train.WorkerSessionCreator.create_session()` {#WorkerSessionCreator.create_session}






## Reading Summaries from Event Files

See [Summaries and
TensorBoard](../../how_tos/summaries_and_tensorboard/index.md) for an
overview of summaries, event files, and visualization in TensorBoard.

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
# with: `tf.summary.scalar('loss', loss_tensor)`.
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



## Training Utilities

- - -

### `tf.train.global_step(sess, global_step_tensor)` {#global_step}

Small helper to get the global step.

```python
# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
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

### `tf.train.basic_train_loop(supervisor, train_step_fn, args=None, kwargs=None, master='')` {#basic_train_loop}

Basic loop to train a model.

Calls `train_step_fn` in a loop to train a model.  The function is called as:

```python
train_step_fn(session, *args, **kwargs)
```

It is passed a `tf.Session` in addition to `args` and `kwargs`.  The function
typically runs one training step in the session.

##### Args:


*  <b>`supervisor`</b>: `tf.Supervisor` to run the training services.
*  <b>`train_step_fn`</b>: Callable to execute one training step.  Called
    repeatedly as `train_step_fn(session, *args **kwargs)`.
*  <b>`args`</b>: Optional positional arguments passed to `train_step_fn`.
*  <b>`kwargs`</b>: Optional keyword arguments passed to `train_step_fn`.
*  <b>`master`</b>: Master to use to create the training session.  Defaults to
    `""` which causes the session to be created in the local process.


- - -

### `tf.train.get_global_step(graph=None)` {#get_global_step}

Get the global step tensor.

The global step tensor must be an integer variable. We first try to find it
in the collection `GLOBAL_STEP`, or by name `global_step:0`.

##### Args:


*  <b>`graph`</b>: The graph to find the global step in. If missing, use default graph.

##### Returns:

  The global step variable, or `None` if none was found.

##### Raises:


*  <b>`TypeError`</b>: If the global step tensor has a non-integer type, or if it is not
    a `Variable`.


- - -

### `tf.train.assert_global_step(global_step_tensor)` {#assert_global_step}

Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

##### Args:


*  <b>`global_step_tensor`</b>: `Tensor` to test.


- - -

### `tf.train.write_graph(graph_or_graph_def, logdir, name, as_text=True)` {#write_graph}

Writes a graph proto to a file.

The graph is written as a binary proto unless `as_text` is `True`.

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

or

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')
```

##### Args:


*  <b>`graph_or_graph_def`</b>: A `Graph` or a `GraphDef` protocol buffer.
*  <b>`logdir`</b>: Directory where to write the graph. This can refer to remote
    filesystems, such as Google Cloud Storage (GCS).
*  <b>`name`</b>: Filename for the graph.
*  <b>`as_text`</b>: If `True`, writes the graph as an ASCII proto.


- - -

### `class tf.train.SessionRunHook` {#SessionRunHook}

Hook to extend calls to MonitoredSession.run().
- - -

#### `tf.train.SessionRunHook.after_create_session(session)` {#SessionRunHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.SessionRunHook.after_run(run_context, run_values)` {#SessionRunHook.after_run}

Called after each call to run().

The `run_values` argument contains results of requested ops/tensors by
`before_run()`.

The `run_context` argument is the same one send to `before_run` call.
`run_context.request_stop()` can be called to stop the iteration.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.
*  <b>`run_values`</b>: A SessionRunValues object.


- - -

#### `tf.train.SessionRunHook.before_run(run_context)` {#SessionRunHook.before_run}

Called before each call to run().

You can return from this call a `SessionRunArgs` object indicating ops or
tensors to add to the upcoming `run()` call.  These ops/tensors will be run
together with the ops/tensors originally passed to the original run() call.
The run args you return can also contain feeds to be added to the run()
call.

The `run_context` argument is a `SessionRunContext` that provides
information about the upcoming `run()` call: the originally requested
op/tensors, the TensorFlow Session.

At this point graph is finalized and you can not add ops.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.

##### Returns:

  None or a `SessionRunArgs` object.


- - -

#### `tf.train.SessionRunHook.begin()` {#SessionRunHook.begin}

Called once before using the session.

When called, the default graph is the one that will be launched in the
session.  The hook can modify the graph by adding new operations to it.
After the `begin()` call the graph will be finalized and the other callbacks
can not modify the graph anymore. Second call of `begin()` on the same
graph, should not change the graph.


- - -

#### `tf.train.SessionRunHook.end(session)` {#SessionRunHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.LoggingTensorHook` {#LoggingTensorHook}

Prints the given tensors once every N local steps or once every N seconds.

The tensors will be printed to the log, with `INFO` severity.
- - -

#### `tf.train.LoggingTensorHook.__init__(tensors, every_n_iter=None, every_n_secs=None)` {#LoggingTensorHook.__init__}

Initializes a LoggingHook monitor.

##### Args:


*  <b>`tensors`</b>: `dict` that maps string-valued tags to tensors/tensor names,
      or `iterable` of tensors/tensor names.
*  <b>`every_n_iter`</b>: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
*  <b>`every_n_secs`</b>: `int` or `float`, print the values of `tensors` once every N
      seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
      provided.

##### Raises:


*  <b>`ValueError`</b>: if `every_n_iter` is non-positive.


- - -

#### `tf.train.LoggingTensorHook.after_create_session(session)` {#LoggingTensorHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.LoggingTensorHook.after_run(run_context, run_values)` {#LoggingTensorHook.after_run}




- - -

#### `tf.train.LoggingTensorHook.before_run(run_context)` {#LoggingTensorHook.before_run}




- - -

#### `tf.train.LoggingTensorHook.begin()` {#LoggingTensorHook.begin}




- - -

#### `tf.train.LoggingTensorHook.end(session)` {#LoggingTensorHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.StopAtStepHook` {#StopAtStepHook}

Monitor to request stop at a specified step.
- - -

#### `tf.train.StopAtStepHook.__init__(num_steps=None, last_step=None)` {#StopAtStepHook.__init__}

Create a StopAtStep Hook.

This hook requests stop after either a number of steps have been
executed or a last step has been reached.  Only of the two options can be
specified.

if `num_steps` is specified, it indicates the number of steps to execute
after `begin()` is called.  If instead `last_step` is specified, it
indicates the last step we want to execute, as passed to the `after_run()`
call.

##### Args:


*  <b>`num_steps`</b>: Number of steps to execute.
*  <b>`last_step`</b>: Step after which to stop.

##### Raises:


*  <b>`ValueError`</b>: If one of the arguments is invalid.


- - -

#### `tf.train.StopAtStepHook.after_create_session(session)` {#StopAtStepHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.StopAtStepHook.after_run(run_context, run_values)` {#StopAtStepHook.after_run}




- - -

#### `tf.train.StopAtStepHook.before_run(run_context)` {#StopAtStepHook.before_run}




- - -

#### `tf.train.StopAtStepHook.begin()` {#StopAtStepHook.begin}




- - -

#### `tf.train.StopAtStepHook.end(session)` {#StopAtStepHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.CheckpointSaverHook` {#CheckpointSaverHook}

Saves checkpoints every N steps or seconds.
- - -

#### `tf.train.CheckpointSaverHook.__init__(checkpoint_dir, save_secs=None, save_steps=None, saver=None, checkpoint_basename='model.ckpt', scaffold=None, listeners=None)` {#CheckpointSaverHook.__init__}

Initialize CheckpointSaverHook monitor.

##### Args:


*  <b>`checkpoint_dir`</b>: `str`, base directory for the checkpoint files.
*  <b>`save_secs`</b>: `int`, save every N secs.
*  <b>`save_steps`</b>: `int`, save every N steps.
*  <b>`saver`</b>: `Saver` object, used for saving.
*  <b>`checkpoint_basename`</b>: `str`, base name for the checkpoint files.
*  <b>`scaffold`</b>: `Scaffold`, use to get saver object.
*  <b>`listeners`</b>: List of `CheckpointSaverListener` subclass instances.
    Used for callbacks that run immediately after the corresponding
    CheckpointSaverHook callbacks, only in steps where the
    CheckpointSaverHook was triggered.

##### Raises:


*  <b>`ValueError`</b>: One of `save_steps` or `save_secs` should be set.
*  <b>`ValueError`</b>: Exactly one of saver or scaffold should be set.


- - -

#### `tf.train.CheckpointSaverHook.after_create_session(session)` {#CheckpointSaverHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.CheckpointSaverHook.after_run(run_context, run_values)` {#CheckpointSaverHook.after_run}




- - -

#### `tf.train.CheckpointSaverHook.before_run(run_context)` {#CheckpointSaverHook.before_run}




- - -

#### `tf.train.CheckpointSaverHook.begin()` {#CheckpointSaverHook.begin}




- - -

#### `tf.train.CheckpointSaverHook.end(session)` {#CheckpointSaverHook.end}





- - -

### `tf.train.NewCheckpointReader(filepattern)` {#NewCheckpointReader}




- - -

### `class tf.train.StepCounterHook` {#StepCounterHook}

Steps per second monitor.
- - -

#### `tf.train.StepCounterHook.__init__(every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None)` {#StepCounterHook.__init__}




- - -

#### `tf.train.StepCounterHook.after_create_session(session)` {#StepCounterHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.StepCounterHook.after_run(run_context, run_values)` {#StepCounterHook.after_run}




- - -

#### `tf.train.StepCounterHook.before_run(run_context)` {#StepCounterHook.before_run}




- - -

#### `tf.train.StepCounterHook.begin()` {#StepCounterHook.begin}




- - -

#### `tf.train.StepCounterHook.end(session)` {#StepCounterHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.NanLossDuringTrainingError` {#NanLossDuringTrainingError}


- - -

#### `tf.train.NanLossDuringTrainingError.__str__()` {#NanLossDuringTrainingError.__str__}





- - -

### `class tf.train.NanTensorHook` {#NanTensorHook}

NaN Loss monitor.

Monitors loss and stops training if loss is NaN.
Can either fail with exception or just stop training.
- - -

#### `tf.train.NanTensorHook.__init__(loss_tensor, fail_on_nan_loss=True)` {#NanTensorHook.__init__}

Initializes NanLoss monitor.

##### Args:


*  <b>`loss_tensor`</b>: `Tensor`, the loss tensor.
*  <b>`fail_on_nan_loss`</b>: `bool`, whether to raise exception when loss is NaN.


- - -

#### `tf.train.NanTensorHook.after_create_session(session)` {#NanTensorHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.NanTensorHook.after_run(run_context, run_values)` {#NanTensorHook.after_run}




- - -

#### `tf.train.NanTensorHook.before_run(run_context)` {#NanTensorHook.before_run}




- - -

#### `tf.train.NanTensorHook.begin()` {#NanTensorHook.begin}

Called once before using the session.

When called, the default graph is the one that will be launched in the
session.  The hook can modify the graph by adding new operations to it.
After the `begin()` call the graph will be finalized and the other callbacks
can not modify the graph anymore. Second call of `begin()` on the same
graph, should not change the graph.


- - -

#### `tf.train.NanTensorHook.end(session)` {#NanTensorHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.SummarySaverHook` {#SummarySaverHook}

Saves summaries every N steps.
- - -

#### `tf.train.SummarySaverHook.__init__(save_steps=None, save_secs=None, output_dir=None, summary_writer=None, scaffold=None, summary_op=None)` {#SummarySaverHook.__init__}

Initializes a `SummarySaver` monitor.

##### Args:


*  <b>`save_steps`</b>: `int`, save summaries every N steps. Exactly one of
      `save_secs` and `save_steps` should be set.
*  <b>`save_secs`</b>: `int`, save summaries every N seconds.
*  <b>`output_dir`</b>: `string`, the directory to save the summaries to. Only used
      if no `summary_writer` is supplied.
*  <b>`summary_writer`</b>: `SummaryWriter`. If `None` and an `output_dir` was passed,
      one will be created accordingly.
*  <b>`scaffold`</b>: `Scaffold` to get summary_op if it's not provided.
*  <b>`summary_op`</b>: `Tensor` of type `string` containing the serialized `Summary`
      protocol buffer or a list of `Tensor`. They are most likely an output
      by TF summary methods like `tf.summary.scalar` or
      `tf.summary.merge_all`. It can be passed in as one tensor; if more
      than one, they must be passed in as a list.

##### Raises:


*  <b>`ValueError`</b>: Exactly one of scaffold or summary_op should be set.


- - -

#### `tf.train.SummarySaverHook.after_create_session(session)` {#SummarySaverHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.SummarySaverHook.after_run(run_context, run_values)` {#SummarySaverHook.after_run}




- - -

#### `tf.train.SummarySaverHook.before_run(run_context)` {#SummarySaverHook.before_run}




- - -

#### `tf.train.SummarySaverHook.begin()` {#SummarySaverHook.begin}




- - -

#### `tf.train.SummarySaverHook.end(session=None)` {#SummarySaverHook.end}





- - -

### `class tf.train.GlobalStepWaiterHook` {#GlobalStepWaiterHook}

Delay execution until global step reaches to wait_until_step.

This hook delays execution until global step reaches to `wait_until_step`. It
is used to gradually start workers in distributed settings. One example usage
would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
task_id=0 is the chief.
- - -

#### `tf.train.GlobalStepWaiterHook.__init__(wait_until_step)` {#GlobalStepWaiterHook.__init__}

Create a _GlobalStepWaiterHook.

##### Args:


*  <b>`wait_until_step`</b>: an `int` shows until which global step should we wait.


- - -

#### `tf.train.GlobalStepWaiterHook.after_create_session(session)` {#GlobalStepWaiterHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.GlobalStepWaiterHook.after_run(run_context, run_values)` {#GlobalStepWaiterHook.after_run}

Called after each call to run().

The `run_values` argument contains results of requested ops/tensors by
`before_run()`.

The `run_context` argument is the same one send to `before_run` call.
`run_context.request_stop()` can be called to stop the iteration.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.
*  <b>`run_values`</b>: A SessionRunValues object.


- - -

#### `tf.train.GlobalStepWaiterHook.before_run(run_context)` {#GlobalStepWaiterHook.before_run}




- - -

#### `tf.train.GlobalStepWaiterHook.begin()` {#GlobalStepWaiterHook.begin}




- - -

#### `tf.train.GlobalStepWaiterHook.end(session)` {#GlobalStepWaiterHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.



- - -

### `class tf.train.SessionRunArgs` {#SessionRunArgs}

Represents arguments to be added to a `Session.run()` call.

Args:
  fetches: Exactly like the 'fetches' argument to Session.Run().
    Can be a single tensor or op, a list of 'fetches' or a dictionary
    of fetches.  For example:
      fetches = global_step_tensor
      fetches = [train_op, summary_op, global_step_tensor]
      fetches = {'step': global_step_tensor, 'summ': summary_op}
    Note that this can recurse as expected:
      fetches = {'step': global_step_tensor,
                 'ops': [train_op, check_nan_op]}
  feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
  options: Exactly like the `options` argument to `Session.run()`, i.e., a
    config_pb2.RunOptions proto.
- - -

#### `tf.train.SessionRunArgs.__getnewargs__()` {#SessionRunArgs.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.train.SessionRunArgs.__getstate__()` {#SessionRunArgs.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.train.SessionRunArgs.__new__(cls, fetches, feed_dict=None, options=None)` {#SessionRunArgs.__new__}




- - -

#### `tf.train.SessionRunArgs.__repr__()` {#SessionRunArgs.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.train.SessionRunArgs.feed_dict` {#SessionRunArgs.feed_dict}

Alias for field number 1


- - -

#### `tf.train.SessionRunArgs.fetches` {#SessionRunArgs.fetches}

Alias for field number 0


- - -

#### `tf.train.SessionRunArgs.options` {#SessionRunArgs.options}

Alias for field number 2



- - -

### `class tf.train.SessionRunContext` {#SessionRunContext}

Provides information about the `session.run()` call being made.

Provides information about original request to `Session.Run()` function.
SessionRunHook objects can stop the loop by calling `request_stop()` of
`run_context`. In the future we may use this object to add more information
about run without changing the Hook API.
- - -

#### `tf.train.SessionRunContext.__init__(original_args, session)` {#SessionRunContext.__init__}

Initializes SessionRunContext.


- - -

#### `tf.train.SessionRunContext.original_args` {#SessionRunContext.original_args}

A `SessionRunArgs` object holding the original arguments of `run()`.

If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
field is equal to SessionRunArgs(a, b).

##### Returns:

 A `SessionRunArgs` object


- - -

#### `tf.train.SessionRunContext.request_stop()` {#SessionRunContext.request_stop}

Sets stop requested field.

Hooks can use this function to request stop of iterations.
`MonitoredSession` checks whether this is called or not.


- - -

#### `tf.train.SessionRunContext.session` {#SessionRunContext.session}

A TensorFlow session object which will execute the `run`.


- - -

#### `tf.train.SessionRunContext.stop_requested` {#SessionRunContext.stop_requested}

Returns whether a stop is requested or not.

If true, `MonitoredSession` stops iterations.

##### Returns:

  A `bool`



- - -

### `class tf.train.SessionRunValues` {#SessionRunValues}

Contains the results of `Session.run()`.

In the future we may use this object to add more information about result of
run without changing the Hook API.

Args:
  results: The return values from `Session.run()` corresponding to the fetches
    attribute returned in the RunArgs. Note that this has the same shape as
    the RunArgs fetches.  For example:
      fetches = global_step_tensor
      => results = nparray(int)
      fetches = [train_op, summary_op, global_step_tensor]
      => results = [None, nparray(string), nparray(int)]
      fetches = {'step': global_step_tensor, 'summ': summary_op}
      => results = {'step': nparray(int), 'summ': nparray(string)}
  options: `RunOptions` from the `Session.run()` call.
  run_metadata: `RunMetadata` from the `Session.run()` call.
- - -

#### `tf.train.SessionRunValues.__getnewargs__()` {#SessionRunValues.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.train.SessionRunValues.__getstate__()` {#SessionRunValues.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.train.SessionRunValues.__new__(_cls, results, options, run_metadata)` {#SessionRunValues.__new__}

Create new instance of SessionRunValues(results, options, run_metadata)


- - -

#### `tf.train.SessionRunValues.__repr__()` {#SessionRunValues.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.train.SessionRunValues.options` {#SessionRunValues.options}

Alias for field number 1


- - -

#### `tf.train.SessionRunValues.results` {#SessionRunValues.results}

Alias for field number 0


- - -

#### `tf.train.SessionRunValues.run_metadata` {#SessionRunValues.run_metadata}

Alias for field number 2



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




## Other Functions and Classes
- - -

### `class tf.train.SyncReplicasOptimizer` {#SyncReplicasOptimizer}

Class to synchronize, aggregate gradients and pass them to the optimizer.

In a typical asynchronous training environment, it's common to have some
stale gradients. For example, with a N-replica asynchronous training,
gradients will be applied to the variables N times independently. Depending
on each replica's training speed, some gradients might be calculated from
copies of the variable from several steps back (N-1 steps on average). This
optimizer avoids stale gradients by collecting gradients from all replicas,
averaging them, then applying them to the variables in one shot, after
which replicas can fetch the new variables and continue.

The following accumulators/queue are created:
<empty line>
* N `gradient accumulators`, one per variable to train. Gradients are pushed
  to them and the chief worker will wait until enough gradients are collected
  and then average them before applying to variables. The accumulator will
  drop all stale gradients (more details in the accumulator op).
* 1 `token` queue where the optimizer pushes the new global_step value after
  all variables are updated.

The following local variable is created:
* `sync_rep_local_step`, one per replica. Compared against the global_step in
  each accumulator to check for staleness of the gradients.

The optimizer adds nodes to the graph to collect gradients and pause the
trainers until variables are updated.
For the Parameter Server job:
<empty line>
1. An accumulator is created for each variable, and each replica pushes the
   gradients into the accumulators instead of directly applying them to the
   variables.
2. Each accumulator averages once enough gradients (replicas_to_aggregate)
   have been accumulated.
3. Apply the averaged gradients to the variables.
4. Only after all variables have been updated, increment the global step.
5. Only after step 4, pushes `global_step` in the `token_queue`, once for
   each worker replica. The workers can now fetch the global step, use it to
   update its local_step variable and start the next batch.

For the replicas:
<empty line>
1. Start a step: fetch variables and compute gradients.
2. Once the gradients have been computed, push them into gradient
   accumulators. Each accumulator will check the staleness and drop the stale.
3. After pushing all the gradients, dequeue an updated value of global_step
   from the token queue and record that step to its local_step variable. Note
   that this is effectively a barrier.
4. Start the next batch.

### Usage

```python
# Create any optimizer to update the variables, say a simple SGD:
opt = GradientDescentOptimizer(learning_rate=0.1)

# Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
# step the optimizer collects 50 gradients before applying to variables.
# Note that if you want to have 2 backup replicas, you can change
# total_num_replicas=52 and make sure this number matches how many physical
# replicas you started in your job.
opt = tf.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
                               total_num_replicas=50)

# Some models have startup_delays to help stabilize the model but when using
# sync_replicas training, set it to 0.

# Now you can call `minimize()` or `compute_gradients()` and
# `apply_gradients()` normally
grads = opt.minimize(total_loss, global_step=self.global_step)


# You can now call get_init_tokens_op() and get_chief_queue_runner().
# Note that get_init_tokens_op() must be called before creating session
# because it modifies the graph by adding new nodes.
init_token_op = opt.get_init_tokens_op()
chief_queue_runner = opt.get_chief_queue_runner()
```

In the training program, every worker will run the train_op as if not
synchronized. But one worker (usually the chief) will need to execute the
chief_queue_runner and get_init_tokens_op from this optimizer.

```python
# When you create the supervisor, you need to add the local_init_op and
# ready_for_local_init_op to make sure the local_step is initialized to the
# global_step. Here is an example:
if is_chief:
  local_init_op = opt.chief_init_op
else:
  local_init_op = opt.local_step_init_op
ready_for_local_init_op = opt.ready_for_local_init_op
sv = tf.Supervisor(graph=g,
                   is_chief=is_chief,
                   # This initialize local step.
                   local_init_op=local_init_op,
                   # This makes sure global step is initialized before using.
                   ready_for_local_init_op=ready_for_local_init_op,
                   saver=model.saver)

# After the session is created by the Supervisor and before the main while
# loop:
if is_chief and FLAGS.sync_replicas:
  sv.start_queue_runners(sess, [chief_queue_runner])
  # Insert initial tokens to the queue.
  sess.run(init_token_op)
```

- - -

#### `tf.train.SyncReplicasOptimizer.__init__(opt, replicas_to_aggregate, total_num_replicas=None, variable_averages=None, variables_to_average=None, use_locking=False, name='sync_replicas')` {#SyncReplicasOptimizer.__init__}

Construct a sync_replicas optimizer.

##### Args:


*  <b>`opt`</b>: The actual optimizer that will be used to compute and apply the
    gradients. Must be one of the Optimizer classes.
*  <b>`replicas_to_aggregate`</b>: number of replicas to aggregate for each variable
    update.
*  <b>`total_num_replicas`</b>: Total number of tasks/workers/replicas, could be
    different from replicas_to_aggregate.
    If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
    replicas_to_aggregate.
    If total_num_replicas < replicas_to_aggregate: Replicas compute
    multiple batches per update to variables.
*  <b>`variable_averages`</b>: Optional `ExponentialMovingAverage` object, used to
    maintain moving averages for the variables passed in
    `variables_to_average`.
*  <b>`variables_to_average`</b>: a list of variables that need to be averaged. Only
    needed if variable_averages is passed in.
*  <b>`use_locking`</b>: If True use locks for update operation.
*  <b>`name`</b>: string. Optional name of the returned operation.


- - -

#### `tf.train.SyncReplicasOptimizer.compute_gradients(*args, **kwargs)` {#SyncReplicasOptimizer.compute_gradients}

Compute gradients of "loss" for the variables in "var_list".

This simply wraps the compute_gradients() from the real optimizer. The
gradients will be aggregated in the apply_gradients() so that user can
modify the gradients like clipping with per replica global norm if needed.
The global norm with aggregated gradients can be bad as one replica's huge
gradients can hurt the gradients from other replicas.

##### Args:


*  <b>`*args`</b>: Arguments for compute_gradients().
*  <b>`**kwargs`</b>: Keyword arguments for compute_gradients().

##### Returns:

  A list of (gradient, variable) pairs.


- - -

#### `tf.train.SyncReplicasOptimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` {#SyncReplicasOptimizer.apply_gradients}

Apply gradients to variables.

This contains most of the synchronization implementation and also wraps the
apply_gradients() from the real optimizer.

##### Args:


*  <b>`grads_and_vars`</b>: List of (gradient, variable) pairs as returned by
    compute_gradients().
*  <b>`global_step`</b>: Optional Variable to increment by one after the
    variables have been updated.
*  <b>`name`</b>: Optional name for the returned operation.  Default to the
    name passed to the Optimizer constructor.

##### Returns:


*  <b>`train_op`</b>: The op to dequeue a token so the replicas can exit this batch
  and start the next one. This is executed by each replica.

##### Raises:


*  <b>`ValueError`</b>: If the grads_and_vars is empty.
*  <b>`ValueError`</b>: If global step is not provided, the staleness cannot be
    checked.


- - -

#### `tf.train.SyncReplicasOptimizer.get_chief_queue_runner()` {#SyncReplicasOptimizer.get_chief_queue_runner}

Returns the QueueRunner for the chief to execute.

This includes the operations to synchronize replicas: aggregate gradients,
apply to variables, increment global step, insert tokens to token queue.

Note that this can only be called after calling apply_gradients() which
actually generates this queuerunner.

##### Returns:

  A `QueueRunner` for chief to execute.

##### Raises:


*  <b>`ValueError`</b>: If this is called before apply_gradients().


- - -

#### `tf.train.SyncReplicasOptimizer.get_init_tokens_op(num_tokens=-1)` {#SyncReplicasOptimizer.get_init_tokens_op}

Returns the op to fill the sync_token_queue with the tokens.

This is supposed to be executed in the beginning of the chief/sync thread
so that even if the total_num_replicas is less than replicas_to_aggregate,
the model can still proceed as the replicas can compute multiple steps per
variable update. Make sure:
`num_tokens >= replicas_to_aggregate - total_num_replicas`.

##### Args:


*  <b>`num_tokens`</b>: Number of tokens to add to the queue.

##### Returns:

  An op for the chief/sync replica to fill the token queue.

##### Raises:


*  <b>`ValueError`</b>: If this is called before apply_gradients().
*  <b>`ValueError`</b>: If num_tokens are smaller than replicas_to_aggregate -
    total_num_replicas.



#### Other Methods
- - -

#### `tf.train.SyncReplicasOptimizer.get_slot(*args, **kwargs)` {#SyncReplicasOptimizer.get_slot}

Return a slot named "name" created for "var" by the Optimizer.

This simply wraps the get_slot() from the actual optimizer.

##### Args:


*  <b>`*args`</b>: Arguments for get_slot().
*  <b>`**kwargs`</b>: Keyword arguments for get_slot().

##### Returns:

  The `Variable` for the slot if it was created, `None` otherwise.


- - -

#### `tf.train.SyncReplicasOptimizer.get_slot_names(*args, **kwargs)` {#SyncReplicasOptimizer.get_slot_names}

Return a list of the names of slots created by the `Optimizer`.

This simply wraps the get_slot_names() from the actual optimizer.

##### Args:


*  <b>`*args`</b>: Arguments for get_slot().
*  <b>`**kwargs`</b>: Keyword arguments for get_slot().

##### Returns:

  A list of strings.



- - -

### `tf.train.checkpoint_exists(checkpoint_prefix)` {#checkpoint_exists}

Checks whether a V1 or V2 checkpoint exists with the specified prefix.

This is the recommended way to check if a checkpoint exists, since it takes
into account the naming difference between V1 and V2 formats.

##### Args:


*  <b>`checkpoint_prefix`</b>: the prefix of a V1 or V2 checkpoint, with V2 taking
    priority.  Typically the result of `Saver.save()` or that of
    `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
    V1/V2.

##### Returns:

  A bool, true iff a checkpoint referred to by `checkpoint_prefix` exists.


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


- - -

### `tf.train.get_checkpoint_mtimes(checkpoint_prefixes)` {#get_checkpoint_mtimes}

Returns the mtimes (modification timestamps) of the checkpoints.

Globs for the checkpoints pointed to by `checkpoint_prefixes`.  If the files
exist, collect their mtime.  Both V2 and V1 checkpoints are considered, in
that priority.

This is the recommended way to get the mtimes, since it takes into account
the naming difference between V1 and V2 formats.

##### Args:


*  <b>`checkpoint_prefixes`</b>: a list of checkpoint paths, typically the results of
    `Saver.save()` or those of `tf.train.latest_checkpoint()`, regardless of
    sharded/non-sharded or V1/V2.

##### Returns:

  A list of mtimes (in microseconds) of the found checkpoints.


