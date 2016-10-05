A conditional accumulator for aggregating gradients.

Up-to-date gradients (i.e., time step at which gradient was computed is
equal to the accumulator's time step) are added to the accumulator.

Extraction of the average gradient is blocked until the required number of
gradients has been accumulated.
- - -

#### `tf.ConditionalAccumulator.__init__(dtype, shape=None, shared_name=None, name='conditional_accumulator')` {#ConditionalAccumulator.__init__}

Creates a new ConditionalAccumulator.

##### Args:


*  <b>`dtype`</b>: Datatype of the accumulated gradients.
*  <b>`shape`</b>: Shape of the accumulated gradients.
*  <b>`shared_name`</b>: Optional. If non-empty, this accumulator will be shared under
    the given name across multiple sessions.
*  <b>`name`</b>: Optional name for the accumulator.


- - -

#### `tf.ConditionalAccumulator.accumulator_ref` {#ConditionalAccumulator.accumulator_ref}

The underlying accumulator reference.


- - -

#### `tf.ConditionalAccumulator.apply_grad(grad, local_step=0, name=None)` {#ConditionalAccumulator.apply_grad}

Attempts to apply a gradient to the accumulator.

The attempt is silently dropped if the gradient is stale, i.e., local_step
is less than the accumulator's global time step.

##### Args:


*  <b>`grad`</b>: The gradient tensor to be applied.
*  <b>`local_step`</b>: Time step at which the gradient was computed.
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  The operation that (conditionally) applies a gradient to the accumulator.

##### Raises:


*  <b>`ValueError`</b>: If grad is of the wrong shape


- - -

#### `tf.ConditionalAccumulator.dtype` {#ConditionalAccumulator.dtype}

The datatype of the gradients accumulated by this accumulator.


- - -

#### `tf.ConditionalAccumulator.name` {#ConditionalAccumulator.name}

The name of the underlying accumulator.


- - -

#### `tf.ConditionalAccumulator.num_accumulated(name=None)` {#ConditionalAccumulator.num_accumulated}

Number of gradients that have currently been aggregated in accumulator.

##### Args:


*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Number of accumulated gradients currently in accumulator.


- - -

#### `tf.ConditionalAccumulator.set_global_step(new_global_step, name=None)` {#ConditionalAccumulator.set_global_step}

Sets the global time step of the accumulator.

The operation logs a warning if we attempt to set to a time step that is
lower than the accumulator's own time step.

##### Args:


*  <b>`new_global_step`</b>: Value of new time step. Can be a variable or a constant
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Operation that sets the accumulator's time step.


- - -

#### `tf.ConditionalAccumulator.take_grad(num_required, name=None)` {#ConditionalAccumulator.take_grad}

Attempts to extract the average gradient from the accumulator.

The operation blocks until sufficient number of gradients have been
successfully applied to the accumulator.

Once successful, the following actions are also triggered:
- Counter of accumulated gradients is reset to 0.
- Aggregated gradient is reset to 0 tensor.
- Accumulator's internal time step is incremented by 1.

##### Args:


*  <b>`num_required`</b>: Number of gradients that needs to have been aggregated
*  <b>`name`</b>: Optional name for the operation

##### Returns:

  A tensor holding the value of the average gradient.

##### Raises:


*  <b>`InvalidArgumentError`</b>: If num_required < 1


