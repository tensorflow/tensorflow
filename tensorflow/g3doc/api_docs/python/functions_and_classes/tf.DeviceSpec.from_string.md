#### `tf.DeviceSpec.from_string(spec)` {#DeviceSpec.from_string}

Construct a `DeviceSpec` from a string.

##### Args:


*  <b>`spec`</b>: a string of the form
   /job:<name>/replica:<id>/task:<id>/device:CPU:<id>
  or
   /job:<name>/replica:<id>/task:<id>/device:GPU:<id>
  as cpu and gpu are mutually exclusive.
  All entries are optional.

##### Returns:

  A DeviceSpec.

