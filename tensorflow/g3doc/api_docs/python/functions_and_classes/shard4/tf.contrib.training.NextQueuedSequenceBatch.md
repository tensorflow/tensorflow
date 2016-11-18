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


