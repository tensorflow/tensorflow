### `tf.contrib.framework.zero_initializer(ref, use_locking=True, name='zero_initializer')` {#zero_initializer}

Initialize 'ref' with all zeros, ref tensor should be uninitialized.
If already initialized, you will get ValueError. This op is intended to
save memory during initialization.

##### Args:


*  <b>`ref`</b>: ref of the tensor need to be zero initialized.
*  <b>`name`</b>: optional name for this operation.

##### Returns:

  ref that initialized.

##### Raises:


*  <b>`ValueError`</b>: If ref tensor is initialized.

