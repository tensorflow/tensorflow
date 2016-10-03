<!-- This file is machine generated: DO NOT EDIT! -->

# Training (contrib)
[TOC]

Training and input utilities.

## Splitting sequence inputs into minibatches with state saving

Use [`SequenceQueueingStateSaver`](#SequenceQueueingStateSaver) or
its wrapper [`batch_sequences_with_states`](#batch_sequences_with_states) if
you have input data with a dynamic primary time / frame count axis which
you'd like to convert into fixed size segments during minibatching, and would
like to store state in the forward direction across segments of an example.

- - -

### `tf.contrib.training.batch_sequences_with_states(input_key, input_sequences, input_context, input_length, initial_states, num_unroll, batch_size, num_threads=3, capacity=1000, allow_small_batch=True, pad=True, name=None)` {#batch_sequences_with_states}

Creates batches of segments of sequential input.

This method creates a `SequenceQueueingStateSaver` (SQSS) and adds it to
the queuerunners. It returns a `NextQueuedSequenceBatch`.

It accepts one example at a time identified by a unique `input_key`.
`input_sequence` is a dict with values that are tensors with time as first
dimension. This time dimension must be the same across those tensors of an
example. It can vary across examples. Although it always has to be a multiple
of `num_unroll`. Hence, padding may be necessary and it is turned on by
default by `pad=True`.

`input_length` is a Tensor scalar or an int recording the time dimension prior
to padding. It should be between 0 and the time dimension. One reason we want
to keep track of it is so that we can take it into consideration when
computing the loss. If `pad=True` then `input_length` can be `None` and will
be inferred.

This methods segments `input_sequence` into segments of length `num_unroll`.
It batches input sequences from `batch_size` many examples. These mini-batches
are available through the `sequence` property of the output. Moreover, for
each entry in the batch we can access its original `input_key` in `key` and
its input length in `total_length`. `length` records within this segment how
many non-padded time steps there are.

Static features of an example that do not vary across time can be part of the
`input_context`, a dict with Tensor values. This method copies the context for
each segment and makes it available in the `context` of the output.

This method can maintain and update a state for each example. It accepts some
initial_states as a dict with Tensor values. The first mini-batch an example
is contained has initial_states as entry of the `state`. If save_state is
called then the next segment will have the updated entry of the `state`.
See `NextQueuedSequenceBatch` for a complete list of properties and methods.

Example usage:

```python
batch_size = 32
num_unroll = 20
num_enqueue_threads = 3
lstm_size = 8
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)

key, sequences, context = my_parser(raw_data)
initial_state_values = tf.zeros((state_size,), dtype=tf.float32)
initial_states = {"lstm_state": initial_state_values}
batch = tf.batch_sequences_with_states(
    input_key=key,
    input_sequences=sequences,
    input_context=context,
    initial_states=initial_states,
    num_unroll=num_unroll,
    batch_size=batch_size,
    num_threads=num_enqueue_threads,
    capacity=batch_size * num_enqueue_threads * 2)

inputs = batch.sequences["input"]
context_label = batch.context["label"]

inputs_by_time = tf.split(1, num_unroll, inputs)
assert len(inputs_by_time) == num_unroll

lstm_output, _ = tf.nn.state_saving_rnn(
  cell,
  inputs_by_time,
  state_saver=batch,
  state_name="lstm_state")

# Start a prefetcher in the background
sess = tf.Session()

tf.train.start_queue_runners(sess=session)

while True:
  # Step through batches, perform training or inference...
  session.run([lstm_output])
```

##### Args:


*  <b>`input_key`</b>: A string scalar `Tensor`, the **unique** key for the given
    input example.  This is used to keep track of the split minibatch elements
    of this input.  Batched keys of the current iteration are made
    accessible via the `key` property.  The shape of `input_key` (scalar) must
    be fully specified.
*  <b>`input_sequences`</b>: A dict mapping string names to `Tensor` values.  The values
    must all have matching first dimension, called `value_length`. They may
    vary from input to input. The remainder of the shape (other than the first
    dimension) must be fully specified.
    The `SequenceQueueingStateSaver` will split these tensors along
    this first dimension into minibatch elements of dimension `num_unrolled`.
    Batched and segmented sequences of the current iteration are made
    accessible via the `sequences` property.

    **Note**: if `pad=False`, then `value_length` must always be a multiple
      of `num_unroll`.

*  <b>`input_context`</b>: A dict mapping string names to `Tensor` values.  The values
    are treated as "global" across all time splits of the given input example,
    and will be copied across for all minibatch elements accordingly.
    Batched and copied context of the current iteration are made
    accessible via the `context` property.

    **Note**: All input_context values must have fully defined shapes.

*  <b>`input_length`</b>: None or an int32 scalar `Tensor`, the length of the sequence
    prior to padding. If `input_length=None` and `pad=True` then the length
    will be inferred and will be equal to `value_length`. If `pad=False` then
    `input_length` cannot be `None`: `input_length` must be specified. Its
    shape of `input_length` (scalar) must be fully specified. Its value may be
    at most `value_length` for any given input (see above for the definition
    of `value_length`). Batched and total lengths of the current iteration are
    made accessible via the `length` and `total_length` properties.
*  <b>`initial_states`</b>: A dict mapping string state names to multi-dimensional
    values (e.g. constants or tensors).  This input defines the set of
    states that will be kept track of during computing iterations, and
    which can be accessed via the `state` and `save_state` methods.

    **Note**: All initial_state values must have fully defined shapes.

*  <b>`num_unroll`</b>: Python integer, how many time steps to unroll at a time.
    The input sequences of length k are then split into k / num_unroll many
    segments.
*  <b>`batch_size`</b>: int or int32 scalar `Tensor`, how large minibatches should
    be when accessing the `state()` method and `context`, `sequences`, etc,
    properties.
*  <b>`num_threads`</b>: The int number of threads enqueuing input examples into a
    queue.
*  <b>`capacity`</b>: The max capacity of the queue in number of examples. Needs to be
    at least `batch_size`. Defaults to 1000. When iterating over the same
    input example multiple times reusing their keys the `capacity` must be
    smaller than the number of examples.
*  <b>`allow_small_batch`</b>: If true, the queue will return smaller batches when
    there aren't enough input examples to fill a whole batch and the end of
    the input has been reached.
*  <b>`pad`</b>: If `True`, `input_sequences` will be padded to multiple of
    `num_unroll`. In that case `input_length` may be `None` and is assumed to
    be the length of first dimension of values in `input_sequences`
    (i.e. `value_length`).
*  <b>`name`</b>: An op name string (optional).

##### Returns:

  A NextQueuedSequenceBatch with segmented and batched inputs and their
  states.

##### Raises:


*  <b>`TypeError`</b>: if any of the inputs is not an expected type.
*  <b>`ValueError`</b>: if any of the input values is inconsistent, e.g. if
    not enough shape information is available from inputs to build
    the state saver.


- - -

### `class tf.contrib.training.NextQueuedSequenceBatch` {#NextQueuedSequenceBatch}

NextQueuedSequenceBatch stores deferred SequenceQueueingStateSaver data.

This class is instantiated by `SequenceQueueingStateSaver` and is accessible
via its `next_batch` property.
- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.__init__(state_saver)` {#NextQueuedSequenceBatch.__init__}




- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.batch_size` {#NextQueuedSequenceBatch.batch_size}

The batch_size of the given batch.

Usually, this is the batch_size requested when initializing the SQSS, but
if allow_small_batch=True this will become smaller when inputs are
exhausted.

##### Returns:

  A scalar integer tensor, the batch_size


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.context` {#NextQueuedSequenceBatch.context}

A dict mapping keys of `input_context` to batched context.

##### Returns:

  A dict mapping keys of `input_context` to tensors.
  If we had at input:

  ```python
  context["name"].get_shape() == [d1, d2, ...]
  ```

  then for this property:

  ```python
  context["name"].get_shape() == [batch_size, d1, d2, ...]
  ```


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.insertion_index` {#NextQueuedSequenceBatch.insertion_index}

The insertion indices of the examples (when they were first added).

These indices start with the value -2**63 and increase with every
call to the prefetch op.  Each whole example gets its own insertion
index, and this is used to prioritize the example so that its truncated
segments appear in adjacent iterations, even if new examples are inserted
by the prefetch op between iterations.

##### Returns:

  An int64 vector of length `batch_size`, the insertion indices.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.key` {#NextQueuedSequenceBatch.key}

The key names of the given truncated unrolled examples.

The format of the key is:

```python
"%05d_of_%05d:%s" % (sequence, sequence_count, original_key)
```

where `original_key` is the unique key read in by the prefetcher.

##### Returns:

  A string vector of length `batch_size`, the keys.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.length` {#NextQueuedSequenceBatch.length}

The lengths of the given truncated unrolled examples.

For initial iterations, for which `sequence * num_unroll < length`,
this number is `num_unroll`.  For the remainder,
this number is between `0` and `num_unroll`.

##### Returns:

  An integer vector of length `batch_size`, the lengths.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.next_key` {#NextQueuedSequenceBatch.next_key}

The key names of the next (in iteration) truncated unrolled examples.

The format of the key is:

```python
"%05d_of_%05d:%s" % (sequence + 1, sequence_count, original_key)
```

if `sequence + 1 < sequence_count`, otherwise:

```python
"STOP:%s" % original_key
```

where `original_key` is the unique key read in by the prefetcher.

##### Returns:

  A string vector of length `batch_size`, the keys.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.save_state(state_name, value, name=None)` {#NextQueuedSequenceBatch.save_state}

Returns an op to save the current batch of state `state_name`.

##### Args:


*  <b>`state_name`</b>: string, matches a key provided in `initial_states`.
*  <b>`value`</b>: A `Tensor`.
    Its type must match that of `initial_states[state_name].dtype`.
    If we had at input:

    ```python
    initial_states[state_name].get_shape() == [d1, d2, ...]
    ```

    then the shape of `value` must match:

    ```python
    tf.shape(value) == [batch_size, d1, d2, ...]
    ```


*  <b>`name`</b>: string (optional).  The name scope for newly created ops.

##### Returns:

  A control flow op that stores the new state of each entry into
  the state saver.  This op must be run for every iteration that
  accesses data from the state saver (otherwise the state saver
  will never progress through its states and run out of capacity).

##### Raises:


*  <b>`KeyError`</b>: if `state_name` does not match any of the initial states
    declared in `initial_states`.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.sequence` {#NextQueuedSequenceBatch.sequence}

An int32 vector, length `batch_size`: the sequence index of each entry.

When an input is split up, the sequence values
```
0, 1, ..., sequence_count - 1
```
are assigned to each split.

##### Returns:

  An int32 vector `Tensor`.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.sequence_count` {#NextQueuedSequenceBatch.sequence_count}

An int32 vector, length `batch_size`: the sequence count of each entry.

When an input is split up, the number of splits is equal to:
`padded_length / num_unroll`.  This is the sequence_count.

##### Returns:

  An int32 vector `Tensor`.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.sequences` {#NextQueuedSequenceBatch.sequences}

A dict mapping keys of `input_sequences` to split and rebatched data.

##### Returns:

  A dict mapping keys of `input_sequences` to tensors.
  If we had at input:

  ```python
  sequences["name"].get_shape() == [None, d1, d2, ...]
  ```

  where `None` meant the sequence time was dynamic, then for this property:

  ```python
  sequences["name"].get_shape() == [batch_size, num_unroll, d1, d2, ...].
  ```


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.state(state_name)` {#NextQueuedSequenceBatch.state}

Returns batched state tensors.

##### Args:


*  <b>`state_name`</b>: string, matches a key provided in `initial_states`.

##### Returns:

  A `Tensor`: a batched set of states, either initial states (if this is
  the first run of the given example), or a value as stored during
  a previous iteration via `save_state` control flow.
  Its type is the same as `initial_states["state_name"].dtype`.
  If we had at input:

  ```python
  initial_states[state_name].get_shape() == [d1, d2, ...],
  ```

  then

  ```python
  state(state_name).get_shape() == [batch_size, d1, d2, ...]
  ```

##### Raises:


*  <b>`KeyError`</b>: if `state_name` does not match any of the initial states
    declared in `initial_states`.


- - -

#### `tf.contrib.training.NextQueuedSequenceBatch.total_length` {#NextQueuedSequenceBatch.total_length}

The lengths of the original (non-truncated) unrolled examples.

##### Returns:

  An integer vector of length `batch_size`, the total lengths.



- - -

### `class tf.contrib.training.SequenceQueueingStateSaver` {#SequenceQueueingStateSaver}

SequenceQueueingStateSaver provides access to stateful values from input.

This class is meant to be used instead of, e.g., a `Queue`, for splitting
variable-length sequence inputs into segments of sequences with fixed length
and batching them into mini-batches.  It maintains contexts and state for a
sequence across the segments.  It can be used in conjunction with a
`QueueRunner` (see the example below).

The `SequenceQueueingStateSaver` (SQSS) accepts one example at a time via the
inputs `input_length`, `input_key`, `input_sequences` (a dict),
`input_context` (a dict), and `initial_states` (a dict).
The sequences, values in `input_sequences`, may have variable first dimension
(the `padded_length`), though this dimension must always be a multiple of
`num_unroll`.  All other dimensions must be fixed and accessible via
`get_shape` calls.  The length prior to padding can be recorded in
`input_length`.  The context values in `input_context` must all have fixed and
well defined dimensions.  The initial state values must all have fixed and
well defined dimensions.

The SQSS splits the sequences of an input example into segments of length
`num_unroll`.  Across examples minibatches of size `batch_size` are formed.
These minibatches contain a segment of the sequences, copy the context values,
and maintain state, length, and key information of the original input
examples.  In the first segment of an example the state is still the initial
state.  It can then be updated; and updated state values are accessible in
subsequent segments of the same example. After each segment
`batch.save_state()` must be called which is done by the state_saving_rnn.
Without this call, the dequeue op associated with the SQSS will not run.
Internally, SQSS has a queue for the input examples. Its `capacity` is
configurable.  If set smaller than `batch_size` then the dequeue op will block
indefinitely.  A small multiple of `batch_size` is a good rule of thumb to
prevent that queue from becoming a bottleneck and slowing down training.
If set too large (and note that it defaults to unbounded) memory consumption
goes up.  Moreover, when iterating over the same input examples multiple times
reusing the same `key` the `capacity` must be smaller than the number of
examples.

The prefetcher, which reads one unrolled, variable-length input sequence at
a time, is accessible via `prefetch_op`.  The underlying `Barrier` object
is accessible via `barrier`.  Processed minibatches, as well as
state read and write capabilities are accessible via `next_batch`.
Specifically, `next_batch` provides access to all of the minibatched
data, including the following, see `NextQueuedSequenceBatch` for details:

*  `total_length`, `length`, `insertion_index`, `key`, `next_key`,
*  `sequence` (the index each minibatch entry's time segment index),
*  `sequence_count` (the total time segment count for each minibatch entry),
*  `context` (a dict of the copied minibatched context values),
*  `sequences` (a dict of the split minibatched variable-length sequences),
*  `state` (to access the states of the current segments of these entries)
*  `save_state` (to save the states for the next segments of these entries)

Example usage:

```python
batch_size = 32
num_unroll = 20
lstm_size = 8
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
initial_state_values = tf.zeros(cell.state_size, dtype=tf.float32)

raw_data = get_single_input_from_input_reader()
length, key, sequences, context = my_parser(raw_data)
assert "input" in sequences.keys()
assert "label" in context.keys()
initial_states = {"lstm_state": initial_state_value}

stateful_reader = tf.SequenceQueueingStateSaver(
    batch_size, num_unroll,
    length=length, input_key=key, input_sequences=sequences,
    input_context=context, initial_states=initial_states,
    capacity=batch_size*100)

batch = stateful_reader.next_batch
inputs = batch.sequences["input"]
context_label = batch.context["label"]

inputs_by_time = tf.split(1, num_unroll, inputs)
assert len(inputs_by_time) == num_unroll

lstm_output, _ = tf.nn.state_saving_rnn(
  cell,
  inputs_by_time,
  state_saver=batch,
  state_name="lstm_state")

# Start a prefetcher in the background
sess = tf.Session()
num_threads = 3
queue_runner = tf.train.QueueRunner(
    stateful_reader, [stateful_reader.prefetch_op] * num_threads)
tf.train.add_queue_runner(queue_runner)
tf.train.start_queue_runners(sess=session)

while True:
  # Step through batches, perform training or inference...
  session.run([lstm_output])
```

**Note**: Usually the barrier is given to a QueueRunner as in the
    examples above.  The QueueRunner will close the barrier if the prefetch_op
    receives an OutOfRange Error from upstream input queues (i.e., reaches
    the end of the input).  If the barrier is closed no further new examples
    are added to the SQSS.  The underlying barrier might, however, still
    contain further unroll-steps of examples that have not undergone all
    iterations.  To gracefully finish all examples, the flag
    `allow_small_batch` must be set to true, which causes the SQSS to issue
    progressively smaller mini-batches with the remaining examples.
- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.__init__(batch_size, num_unroll, input_length, input_key, input_sequences, input_context, initial_states, capacity=None, allow_small_batch=False, name=None)` {#SequenceQueueingStateSaver.__init__}

Creates the SequenceQueueingStateSaver.

##### Args:


*  <b>`batch_size`</b>: int or int32 scalar `Tensor`, how large minibatches should
    be when accessing the `state()` method and `context`, `sequences`, etc,
    properties.
*  <b>`num_unroll`</b>: Python integer, how many time steps to unroll at a time.
    The input sequences of length `k` are then split into `k / num_unroll`
    many segments.
*  <b>`input_length`</b>: An int32 scalar `Tensor`, the length of the sequence prior
    to padding.  This value may be at most `padded_length` for any given
    input (see below for the definition of `padded_length`).
    Batched and total lengths of the current iteration are made accessible
    via the `length` and `total_length` properties.  The shape of
    input_length (scalar) must be fully specified.
*  <b>`input_key`</b>: A string scalar `Tensor`, the **unique** key for the given
    input.  This is used to keep track of the split minibatch elements
    of this input.  Batched keys of the current iteration are made
    accessible via the `key` property.  The shape of `input_key` (scalar)
    must be fully specified.
*  <b>`input_sequences`</b>: A dict mapping string names to `Tensor` values.  The
    values must all have matching first dimension, called `padded_length`.
    The `SequenceQueueingStateSaver` will split these tensors along
    this first dimension into minibatch elements of dimension
    `num_unroll`. Batched and segmented sequences of the current iteration
    are made accessible via the `sequences` property.

    **Note**: `padded_length` may be dynamic, and may vary from input
    to input, but must always be a multiple of `num_unroll`.  The remainder
    of the shape (other than the first dimension) must be fully specified.

*  <b>`input_context`</b>: A dict mapping string names to `Tensor` values.  The values
    are treated as "global" across all time splits of the given input,
    and will be copied across for all minibatch elements accordingly.
    Batched and copied context of the current iteration are made
    accessible via the `context` property.

    **Note**: All input_context values must have fully defined shapes.

*  <b>`initial_states`</b>: A dict mapping string state names to multi-dimensional
    values (e.g. constants or tensors).  This input defines the set of
    states that will be kept track of during computing iterations, and
    which can be accessed via the `state` and `save_state` methods.

    **Note**: All initial_state values must have fully defined shapes.

*  <b>`capacity`</b>: The max capacity of the SQSS in number of examples. Needs to be
    at least `batch_size`. Defaults to unbounded.
*  <b>`allow_small_batch`</b>: If true, the SQSS will return smaller batches when
    there aren't enough input examples to fill a whole batch and the end of
    the input has been reached (i.e., the underlying barrier has been
    closed).
*  <b>`name`</b>: An op name string (optional).

##### Raises:


*  <b>`TypeError`</b>: if any of the inputs is not an expected type.
*  <b>`ValueError`</b>: if any of the input values is inconsistent, e.g. if
  not enough shape information is available from inputs to build
  the state saver.


- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.barrier` {#SequenceQueueingStateSaver.barrier}




- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.batch_size` {#SequenceQueueingStateSaver.batch_size}




- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.close(cancel_pending_enqueues=False, name=None)` {#SequenceQueueingStateSaver.close}

Closes the barrier and the FIFOQueue.

This operation signals that no more segments of new sequences will be
enqueued. New segments of already inserted sequences may still be enqueued
and dequeued if there is a sufficient number filling a batch or
allow_small_batch is true. Otherwise dequeue operations will fail
immediately.

##### Args:


*  <b>`cancel_pending_enqueues`</b>: (Optional.) A boolean, defaulting to
    `False`. If `True`, all pending enqueues to the underlying queues will
    be cancelled, and completing already started sequences is not possible.
*  <b>`name`</b>: Optional name for the op.

##### Returns:

  The operation that closes the barrier and the FIFOQueue.


- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.name` {#SequenceQueueingStateSaver.name}




- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.next_batch` {#SequenceQueueingStateSaver.next_batch}

The `NextQueuedSequenceBatch` providing access to batched output data.

Also provides access to the `state` and `save_state` methods.
The first time this gets called, it additionally prepares barrier reads
and creates `NextQueuedSequenceBatch` / next_batch objects. Subsequent
calls simply return the previously created `next_batch`.

In order to access data in `next_batch` without blocking, the `prefetch_op`
must have been run at least `batch_size` times (ideally in a separate
thread, or launched via a `QueueRunner`). After processing a segment in
`next_batch()`, `batch.save_state()` must be called which is done by the
state_saving_rnn. Without this call, the dequeue op associated with the SQSS
will not run.

##### Returns:

  A cached `NextQueuedSequenceBatch` instance.


- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.num_unroll` {#SequenceQueueingStateSaver.num_unroll}




- - -

#### `tf.contrib.training.SequenceQueueingStateSaver.prefetch_op` {#SequenceQueueingStateSaver.prefetch_op}

The op used to prefetch new data into the state saver.

Running it once enqueues one new input example into the state saver.
The first time this gets called, it additionally creates the prefetch_op.
Subsequent calls simply return the previously created `prefetch_op`.

It should be run in a separate thread via e.g. a `QueueRunner`.

##### Returns:

  An `Operation` that performs prefetching.





## Online data resampling

To resample data with replacement on a per-example basis, use
['resample_at_rate'](#resample_at_rate), providing the desired rate
for each example. If you wish to specify relative rates, rather than
absolute ones, use ['weighted_resample'](#weighted_resample) (which
also returns the actual resampling rate used for each output example).

Use ['stratified_sample'](#stratified_sample) or
['stratified_sample_unknown_dist'](#stratified_sample_unknown_dist) to
resample without replacement from the data to achieve a desired mix of
class proportions that the Tensorflow graph sees. For instance, if you
have a binary classification dataset that is 99.9% class 1, a common
approach is to resample from the data so that the data is more
balanced.

- - -

### `tf.contrib.training.resample_at_rate(inputs, rates, scope=None, seed=None, back_prop=False)` {#resample_at_rate}

Given `inputs` tensors, stochastically resamples each at a given rate.

For example, if the inputs are `[[a1, a2], [b1, b2]]` and the rates
tensor contains `[3, 1]`, then the return value may look like `[[a1,
a2, a1, a1], [b1, b2, b1, b1]]`. However, many other outputs are
possible, since this is stochastic -- averaged over many repeated
calls, each set of inputs should appear in the output `rate` times
the number of invocations.

Uses Knuth's method to generate samples from the poisson
distribution (but instead of just incrementing a count, actually
emits the input); this is described at
https://en.wikipedia.org/wiki/Poisson_distribution in the section on
generating Poisson-distributed random variables.

Note that this method is not appropriate for large rate values: with
float16 it will stop performing correctly for rates above 9.17;
float32, 87; and float64, 708. (These are the base-e versions of the
minimum representable exponent for each type.)

##### Args:


*  <b>`inputs`</b>: A list of tensors, each of which has a shape of `[batch_size, ...]`
*  <b>`rates`</b>: A tensor of shape `[batch_size]` contiaining the resampling rates
         for each input.
*  <b>`scope`</b>: Scope for the op.
*  <b>`seed`</b>: Random seed to use.
*  <b>`back_prop`</b>: Whether to allow back-propagation through this op.

##### Returns:

  Selections from the input tensors.


- - -

### `tf.contrib.training.stratified_sample(tensors, labels, target_probs, batch_size, init_probs=None, enqueue_many=False, queue_capacity=16, threads_per_queue=1, name=None)` {#stratified_sample}

Stochastically creates batches based on per-class probabilities.

This method discards examples. Internally, it creates one queue to amortize
the cost of disk reads, and one queue to hold the properly-proportioned
batch. See `stratified_sample_unknown_dist` for a function that performs
stratified sampling with one queue per class and doesn't require knowing the
class data-distribution ahead of time.

##### Args:


*  <b>`tensors`</b>: List of tensors for data. All tensors are either one item or a
      batch, according to enqueue_many.
*  <b>`labels`</b>: Tensor for label of data. Label is a single integer or a batch,
      depending on enqueue_many. It is not a one-hot vector.
*  <b>`target_probs`</b>: Target class proportions in batch. An object whose type has a
      registered Tensor conversion function.
*  <b>`batch_size`</b>: Size of batch to be returned.
*  <b>`init_probs`</b>: Class proportions in the data. An object whose type has a
      registered Tensor conversion function, or `None` for estimating the
      initial distribution.
*  <b>`enqueue_many`</b>: Bool. If true, interpret input tensors as having a batch
      dimension.
*  <b>`queue_capacity`</b>: Capacity of the large queue that holds input examples.
*  <b>`threads_per_queue`</b>: Number of threads for the large queue that holds input
      examples and for the final queue with the proper class proportions.
*  <b>`name`</b>: Optional prefix for ops created by this function.

##### Raises:


*  <b>`ValueError`</b>: enqueue_many is True and labels doesn't have a batch
      dimension, or if enqueue_many is False and labels isn't a scalar.
*  <b>`ValueError`</b>: enqueue_many is True, and batch dimension on data and labels
      don't match.
*  <b>`ValueError`</b>: if probs don't sum to one.
*  <b>`ValueError`</b>: if a zero initial probability class has a nonzero target
      probability.
*  <b>`TFAssertion`</b>: if labels aren't integers in [0, num classes).

##### Returns:

  (data_batch, label_batch), where data_batch is a list of tensors of the same
      length as `tensors`

##### Example:

  # Get tensor for a single data and label example.
  data, label = data_provider.Get(['data', 'label'])

  # Get stratified batch according to per-class probabilities.
  target_probs = [...distribution you want...]
  [data_batch], labels = tf.contrib.training.stratified_sample(
      [data], label, target_probs)

  # Run batch through network.
  ...


- - -

### `tf.contrib.training.stratified_sample_unknown_dist(tensors, labels, probs, batch_size, enqueue_many=False, queue_capacity=16, threads_per_queue=1, name=None)` {#stratified_sample_unknown_dist}

Stochastically creates batches based on per-class probabilities.

**NOTICE** This sampler can be significantly slower than `stratified_sample`
due to each thread discarding all examples not in its assigned class.

This uses a number of threads proportional to the number of classes. See
`stratified_sample` for an implementation that discards fewer examples and
uses a fixed number of threads. This function's only advantage over
`stratified_sample` is that the class data-distribution doesn't need to be
known ahead of time.

##### Args:


*  <b>`tensors`</b>: List of tensors for data. All tensors are either one item or a
      batch, according to enqueue_many.
*  <b>`labels`</b>: Tensor for label of data. Label is a single integer or a batch,
      depending on enqueue_many. It is not a one-hot vector.
*  <b>`probs`</b>: Target class probabilities. An object whose type has a registered
      Tensor conversion function.
*  <b>`batch_size`</b>: Size of batch to be returned.
*  <b>`enqueue_many`</b>: Bool. If true, interpret input tensors as having a batch
      dimension.
*  <b>`queue_capacity`</b>: Capacity of each per-class queue.
*  <b>`threads_per_queue`</b>: Number of threads for each per-class queue.
*  <b>`name`</b>: Optional prefix for ops created by this function.

##### Raises:


*  <b>`ValueError`</b>: enqueue_many is True and labels doesn't have a batch
      dimension, or if enqueue_many is False and labels isn't a scalar.
*  <b>`ValueError`</b>: enqueue_many is True, and batch dimension of data and labels
      don't match.
*  <b>`ValueError`</b>: if probs don't sum to one.
*  <b>`TFAssertion`</b>: if labels aren't integers in [0, num classes).

##### Returns:

  (data_batch, label_batch), where data_batch is a list of tensors of the same
      length as `tensors`

##### Example:

  # Get tensor for a single data and label example.
  data, label = data_provider.Get(['data', 'label'])

  # Get stratified batch according to per-class probabilities.
  init_probs = [1.0/NUM_CLASSES for _ in range(NUM_CLASSES)]
  [data_batch], labels = (
      tf.contrib.training.stratified_sample_unknown_dist(
          [data], label, init_probs, 16))

  # Run batch through network.
  ...


- - -

### `tf.contrib.training.weighted_resample(inputs, weights, overall_rate, scope=None, mean_decay=0.999, warmup=10, seed=None)` {#weighted_resample}

Performs an approximate weighted resampling of `inputs`.

This method chooses elements from `inputs` where each item's rate of
selection is proportional to its value in `weights`, and the average
rate of selection across all inputs (and many invocations!) is
`overall_rate`.

##### Args:


*  <b>`inputs`</b>: A list of tensors whose first dimension is `batch_size`.
*  <b>`weights`</b>: A `[batch_size]`-shaped tensor with each batch member's weight.
*  <b>`overall_rate`</b>: Desired overall rate of resampling.
*  <b>`scope`</b>: Scope to use for the op.
*  <b>`mean_decay`</b>: How quickly to decay the running estimate of the mean weight.
*  <b>`warmup`</b>: Until the resulting tensor has been evaluated `warmup`
    times, the resampling menthod uses the true mean over all calls
    as its weight estimate, rather than a decayed mean.
*  <b>`seed`</b>: Random seed.

##### Returns:

  A list of tensors exactly like `inputs`, but with an unknown (and
    possibly zero) first dimension.
  A tensor containing the effective resampling rate used for each output.



## Bucketing

Use ['bucket'](#bucket) or
['bucket_by_sequence_length'](#bucket_by_sequence_length) to stratify
minibatches into groups ("buckets").  Use `bucket_by_sequence_length`
with the argument `dynamic_pad=True` to receive minibatches of similarly
sized sequences for efficient training via `dynamic_rnn`.

- - -

### `tf.contrib.training.bucket(tensors, which_bucket, batch_size, num_buckets, num_threads=1, capacity=32, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, keep_input=None, shared_name=None, name=None)` {#bucket}

Lazy bucketing of input tensors according to `which_bucket`.

The argument `tensors` can be a list or a dictionary of tensors.
The value returned by the function will be of the same type
as `tensors`.

The tensors entering this function are put into the bucket given by
`which_bucket`.  Each bucket has its own queue.  When a bucket contains
`batch_size` elements, this minibatch is pushed onto a top queue.  The
tensors returned from this function are a the result of dequeueing the
next minibatch from this top queue.

This function is implemented using several queues. A `QueueRunner` for the
queues is added to the current `Graph`'s `QUEUE_RUNNER` collection.

As the returned tensors are the result of of a dequeue operation, evaluating
them will throw a `tf.errors.OutOfRangeError` when the input queue is
exhausted.  If these tensors are feeding another input queue, its queue runner
will catch this exception, however, if they are used in your main thread
you are responsible for catching this yourself.

*N.B.:* If `dynamic_pad` is `False`, you must ensure that either
(i) the `shapes` argument is passed, or (ii) all of the tensors in
`tensors` must have fully-defined shapes. `ValueError` will be
raised if neither of these conditions holds.

If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
tensors is known, but individual dimensions may have shape `None`.
In this case, for each enqueue the dimensions with value `None`
may have a variable length; upon dequeue, the output tensors will be padded
on the right to the maximum shape of the tensors in the current minibatch.
For numbers, this padding takes value 0.  For strings, this padding is
the empty string.  See `PaddingFIFOQueue` for more info.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queues are closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`get_shape()` method will have a 0th `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

##### Args:


*  <b>`tensors`</b>: The list or dictionary of tensors, representing a single element,
    to bucket.  Nested lists are not supported.
*  <b>`which_bucket`</b>: An `int32` scalar Tensor taking a value in `[0, num_buckets)`.
*  <b>`batch_size`</b>: The new batch size pulled from the queue
    (python int or int32 scalar).
*  <b>`num_buckets`</b>: A python integer, the number of buckets.
*  <b>`num_threads`</b>: An integer.  The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of minibatches in the top queue,
    and also the maximum number of elements within each bucket.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batches to be smaller if there are insufficient items left in the queues.
*  <b>`keep_input`</b>: (Optional).  A `bool` scalar Tensor.  If provided, this tensor
    controls whether the input is added to the queue or not.  If it evaluates
    `True`, then `tensors` are added to the bucket; otherwise they are
    dropped.  This tensor essentially acts as a filtering mechanism.
    The default behavior is to assume `keep_input=True`.
*  <b>`shared_name`</b>: (Optional). If set, the queues will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A tuple `(bucket, outputs)` where `bucket` is
  a `int32` scalar tensor and `outputs` is a list or
  dictionary of batched outputs corresponding to elements of `tensors`.
  Every step will receive a new bucket of outputs.

##### Raises:


*  <b>`ValueError`</b>: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.


- - -

### `tf.contrib.training.bucket_by_sequence_length(input_length, tensors, batch_size, bucket_boundaries, num_threads=1, capacity=32, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, keep_input=None, shared_name=None, name=None)` {#bucket_by_sequence_length}

Lazy bucketing of inputs according to their length.

This method calls `tf.contrib.training.bucket` under the hood, after first
subdividing the bucket boundaries into separate buckets and identifying which
bucket the given `input_length` belongs to.  See the documentation for
`which_bucket` for details of the other arguments.

##### Args:


*  <b>`input_length`</b>: `int32` scalar `Tensor`, the sequence length of tensors.
*  <b>`tensors`</b>: The list or dictionary of tensors, representing a single element,
    to bucket.  Nested lists are not supported.
*  <b>`batch_size`</b>: The new batch size pulled from the queue
    (python int or int32 scalar).
*  <b>`bucket_boundaries`</b>: int list, increasing non-negative numbers.
    The edges of the buckets to use when bucketing tensors.  Two extra buckets
    are created, one for `input_length < bucket_boundaries[0]` and
    one for `input_length >= bucket_boundaries[-1]`.
*  <b>`num_threads`</b>: An integer.  The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of minibatches in the top queue,
    and also the maximum number of elements within each bucket.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batches to be smaller if there are insufficient items left in the queues.
*  <b>`keep_input`</b>: (Optional).  A `bool` scalar Tensor.  If provided, this tensor
    controls whether the input is added to the queue or not.  If it evaluates
    `True`, then `tensors` are added to the bucket; otherwise they are
    dropped.  This tensor essentially acts as a filtering mechanism.
    The default behavior is to assume `keep_input=True`.
*  <b>`shared_name`</b>: (Optional). If set, the queues will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A tuple `(sequence_length, outputs)` where `sequence_length` is
  a 1-D `Tensor` of size `batch_size` and `outputs` is a list or dictionary
  of batched, bucketed, outputs corresponding to elements of `tensors`.

##### Raises:


*  <b>`TypeError`</b>: if `bucket_boundaries` is not a list of python integers.
*  <b>`ValueError`</b>: if `bucket_boundaries` is empty or contains non-increasing
    values.


