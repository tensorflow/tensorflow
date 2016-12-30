A conditional accumulator for aggregating gradients.

Up-to-date gradients (i.e., time step at which gradient was computed is
equal to the accumulator's time step) are added to the accumulator.

Extraction of the average gradient is blocked until the required number of
gradients has been accumulated.
- - -

#### `tf.ConditionalAccumulatorBase.__init__(dtype, shape, accumulator_ref)` {#ConditionalAccumulatorBase.__init__}

Creates a new ConditionalAccumulator.

##### Args:


*  <b>`dtype`</b>: Datatype of the accumulated gradients.
*  <b>`shape`</b>: Shape of the accumulated gradients.
*  <b>`accumulator_ref`</b>: A handle to the conditional accumulator, created by sub-
    classes


- - -

#### `tf.ConditionalAccumulatorBase.accumulator_ref` {#ConditionalAccumulatorBase.accumulator_ref}

The underlying accumulator reference.


- - -

#### `tf.ConditionalAccumulatorBase.dtype` {#ConditionalAccumulatorBase.dtype}

The datatype of the gradients accumulated by this accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.name` {#ConditionalAccumulatorBase.name}

The name of the underlying accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.num_accumulated(name=None)` {#ConditionalAccumulatorBase.num_accumulated}

Number of gradients that have currently been aggregated in accumulator.

##### Args:


*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Number of accumulated gradients currently in accumulator.


- - -

#### `tf.ConditionalAccumulatorBase.set_global_step(new_global_step, name=None)` {#ConditionalAccumulatorBase.set_global_step}

Sets the global time step of the accumulator.

The operation logs a warning if we attempt to set to a time step that is
lower than the accumulator's own time step.

##### Args:


*  <b>`new_global_step`</b>: Value of new time step. Can be a variable or a constant
*  <b>`name`</b>: Optional name for the operation.

##### Returns:

  Operation that sets the accumulator's time step.


