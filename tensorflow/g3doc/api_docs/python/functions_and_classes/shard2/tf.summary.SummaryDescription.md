
- - -

#### `tf.summary.SummaryDescription.ByteSize()` {#SummaryDescription.ByteSize}




- - -

#### `tf.summary.SummaryDescription.Clear()` {#SummaryDescription.Clear}




- - -

#### `tf.summary.SummaryDescription.ClearExtension(extension_handle)` {#SummaryDescription.ClearExtension}




- - -

#### `tf.summary.SummaryDescription.ClearField(field_name)` {#SummaryDescription.ClearField}




- - -

#### `tf.summary.SummaryDescription.CopyFrom(other_msg)` {#SummaryDescription.CopyFrom}

Copies the content of the specified message into the current message.

The method clears the current message and then merges the specified
message using MergeFrom.

##### Args:


*  <b>`other_msg`</b>: Message to copy into the current one.


- - -

#### `tf.summary.SummaryDescription.DiscardUnknownFields()` {#SummaryDescription.DiscardUnknownFields}




- - -

#### `tf.summary.SummaryDescription.FindInitializationErrors()` {#SummaryDescription.FindInitializationErrors}

Finds required fields which are not initialized.

##### Returns:

  A list of strings.  Each string is a path to an uninitialized field from
  the top-level message, e.g. "foo.bar[5].baz".


- - -

#### `tf.summary.SummaryDescription.FromString(s)` {#SummaryDescription.FromString}




- - -

#### `tf.summary.SummaryDescription.HasExtension(extension_handle)` {#SummaryDescription.HasExtension}




- - -

#### `tf.summary.SummaryDescription.HasField(field_name)` {#SummaryDescription.HasField}




- - -

#### `tf.summary.SummaryDescription.IsInitialized(errors=None)` {#SummaryDescription.IsInitialized}

Checks if all required fields of a message are set.

##### Args:


*  <b>`errors`</b>: A list which, if provided, will be populated with the field
           paths of all missing required fields.

##### Returns:

  True iff the specified message has all required fields set.


- - -

#### `tf.summary.SummaryDescription.ListFields()` {#SummaryDescription.ListFields}




- - -

#### `tf.summary.SummaryDescription.MergeFrom(msg)` {#SummaryDescription.MergeFrom}




- - -

#### `tf.summary.SummaryDescription.MergeFromString(serialized)` {#SummaryDescription.MergeFromString}




- - -

#### `tf.summary.SummaryDescription.ParseFromString(serialized)` {#SummaryDescription.ParseFromString}

Parse serialized protocol buffer data into this message.

Like MergeFromString(), except we clear the object first and
do not return the value that MergeFromString returns.


- - -

#### `tf.summary.SummaryDescription.RegisterExtension(extension_handle)` {#SummaryDescription.RegisterExtension}




- - -

#### `tf.summary.SummaryDescription.SerializePartialToString()` {#SummaryDescription.SerializePartialToString}




- - -

#### `tf.summary.SummaryDescription.SerializeToString()` {#SummaryDescription.SerializeToString}




- - -

#### `tf.summary.SummaryDescription.SetInParent()` {#SummaryDescription.SetInParent}

Sets the _cached_byte_size_dirty bit to true,
and propagates this to our listener iff this was a state change.


- - -

#### `tf.summary.SummaryDescription.WhichOneof(oneof_name)` {#SummaryDescription.WhichOneof}

Returns the name of the currently set field inside a oneof, or None.


- - -

#### `tf.summary.SummaryDescription.__deepcopy__(memo=None)` {#SummaryDescription.__deepcopy__}




- - -

#### `tf.summary.SummaryDescription.__eq__(other)` {#SummaryDescription.__eq__}




- - -

#### `tf.summary.SummaryDescription.__getstate__()` {#SummaryDescription.__getstate__}

Support the pickle protocol.


- - -

#### `tf.summary.SummaryDescription.__hash__()` {#SummaryDescription.__hash__}




- - -

#### `tf.summary.SummaryDescription.__init__(**kwargs)` {#SummaryDescription.__init__}




- - -

#### `tf.summary.SummaryDescription.__ne__(other_msg)` {#SummaryDescription.__ne__}




- - -

#### `tf.summary.SummaryDescription.__repr__()` {#SummaryDescription.__repr__}




- - -

#### `tf.summary.SummaryDescription.__setstate__(state)` {#SummaryDescription.__setstate__}

Support the pickle protocol.


- - -

#### `tf.summary.SummaryDescription.__str__()` {#SummaryDescription.__str__}




- - -

#### `tf.summary.SummaryDescription.__unicode__()` {#SummaryDescription.__unicode__}




- - -

#### `tf.summary.SummaryDescription.type_hint` {#SummaryDescription.type_hint}

Magic attribute generated for "type_hint" proto field.


