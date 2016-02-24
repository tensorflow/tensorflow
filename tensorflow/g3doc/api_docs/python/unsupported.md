<!-- This file is machine generated: DO NOT EDIT! -->

# Unsupported
[TOC]

This module includes unsupported and experimental features which are exposed
but not part of the supported public API.  Anything in this module can change
without notice, even across a patch release.

## Utilities

- - -

### `tf.unsupported.constant_value(tensor)` {#constant_value}

Returns the constant value of the given tensor, if efficiently calculable.

This function attempts to partially evaluate the given tensor, and
returns its value as a numpy ndarray if this succeeds.

TODO(mrry): Consider whether this function should use a registration
mechanism like gradients and ShapeFunctions, so that it is easily
extensible.

##### Args:


*  <b>`tensor`</b>: The Tensor to be evaluated.

##### Returns:

  A numpy ndarray containing the constant value of the given `tensor`,
  or None if it cannot be calculated.

##### Raises:


*  <b>`TypeError`</b>: if tensor is not an ops.Tensor.



## Other Functions and Classes
- - -

### `tf.unsupported.make_all(module_name)` {#make_all}

Generate `__all__` from a module's docstring.

Usage: `make_all(__name__)`.  The caller module must have a docstring,
and `__all__` will contain all symbols with `@@` references.

##### Args:


*  <b>`module_name`</b>: The name of the module (usually `__name__`).

##### Returns:

  A list suitable for use as `__all__`.


