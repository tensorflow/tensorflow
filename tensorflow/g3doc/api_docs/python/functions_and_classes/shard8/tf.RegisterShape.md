A decorator for registering the shape function for an op type.

Soon to be removed.  Shape functions should be registered via
the SetShapeFn on the original Op specification in C++.
- - -

#### `tf.RegisterShape.__call__(f)` {#RegisterShape.__call__}

Registers "f" as the shape function for "op_type".


- - -

#### `tf.RegisterShape.__init__(op_type)` {#RegisterShape.__init__}

Saves the `op_type` as the `Operation` type.


