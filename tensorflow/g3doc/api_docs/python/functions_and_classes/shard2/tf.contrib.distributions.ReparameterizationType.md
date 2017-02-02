Instances of this class represent how sampling is reparameterized.

Two static instances exist in the distritributions library, signifying
one of two possible properties for samples from a distribution:

`FULLY_REPARAMETERIZED`: Samples from the distribution are fully
  reparameterized, and straight-through gradients are supported.

`NOT_REPARAMETERIZED`: Samples from the distribution are not fully
  reparameterized, and straight-through gradients are either partially
  unsupported or are not supported at all.  In this case, for purposes of
  e.g. RL or variational inference, it is generally safest to wrap the
  sample results in a `stop_gradients` call and instead use policy
  gradients / surrogate loss instead.
- - -

#### `tf.contrib.distributions.ReparameterizationType.__eq__(other)` {#ReparameterizationType.__eq__}

Determine if this `ReparameterizationType` is equal to another.

Since RepaparameterizationType instances are constant static global
instances, equality checks if two instances' id() values are equal.

##### Args:


*  <b>`other`</b>: Object to compare against.

##### Returns:

  `self is other`.


- - -

#### `tf.contrib.distributions.ReparameterizationType.__init__(rep_type)` {#ReparameterizationType.__init__}




- - -

#### `tf.contrib.distributions.ReparameterizationType.__repr__()` {#ReparameterizationType.__repr__}




