# Aliasing in XLA

This document describes the aliasing API for XLA: when building an XLA program,
you can specify the desired aliasing between the input and output buffers.

## Defining aliasing at compile-time

For example, consider a trivial HLO module which simply adds `1` to its input:

```
HloModule increment

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

This module will allocate two 4-byte buffers: one for the input `%p`, and one
for the output `%out`.

However, it is often desirable to perform the update in-place (for example, if
in the frontend generating the expression the input variable is no longer alive
after the computation, as in the increment `p++`).

To perform such an update efficiently, you can specify the input aliasing:

```
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

The format specifies that the entire output (marked by `{}`) is aliased to the
input parameter `0`.

See the
[`XlaBuilder::SetUpAlias`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)
API to specify the aliasing programmatically.

## Defining aliasing at run-time

The aliasing defined in the previous step is specified during the _compilation_.
During the execution, you can choose whether actually to donate the buffer using
the
[`LocalClient::RunAsync`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/local_client.h)
API.

Input buffers to the program are wrapped in
[`ExecutionInput`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h),
which in turn contain a tree of `MaybeOwningDeviceMemory`. If memory is
specified as _owning_ (ownership of the buffer is passed to the XLA runtime),
the buffer is actually donated, and the update is executed in-place, as
requested by the compile-time aliasing API.

If, however, the buffer which is aliased at compile time is _not_ donated at
runtime, _copy-protection_ kicks in: an extra output buffer `O` is allocated,
and the contents of the input buffer `P` which was meant to be aliased are
copied into `O` (so effectively the program can execute as if the buffer `O` was
donated at runtime).

## Frontend interop

### TF/XLA

In clusters of TensorFlow program compiled with XLA, all resource variable
updates are aliased at compile time (the aliasing at runtime depends on whether
anything else holds a reference to the resource variable tensor).
