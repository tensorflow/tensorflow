# Aliasing in XLA

This document describes the XLA aliasing API, which lets you specify the
aliasing between the input and output buffers when building an XLA program.

## Defining aliasing at compile-time

For example, consider a trivial HLO module which simply adds `1` to its input:

```mlir
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

```mlir
HloModule increment, input_output_alias={ {}: 0 }

ENTRY entry {
  %p = f32[] parameter(0)
  %c = f32[] constant(1)
  ROOT %out = f32[] add(%p, %c)
}
```

The format specifies that the entire output (marked by `{}`) is aliased to the
input parameter `0`.

To specify the aliasing programmatically, see the
[`XlaBuilder::SetUpAlias`](https://github.com/openxla/xla/blob/main/xla/client/xla_builder.h)
API.

## Defining aliasing at runtime

The aliasing defined in the previous step is specified during *compilation*.
During execution, you can use the
[`LocalClient::RunAsync`](https://github.com/openxla/xla/blob/main/xla/client/local_client.h)
API to choose whether to donate the buffer.

Input buffers to the program are wrapped in
[`ExecutionInput`](https://github.com/openxla/xla/blob/main/xla/service/executable.h)s,
which in turn contain a tree of `MaybeOwningDeviceMemory`. If memory is
specified as *owning* (ownership of the buffer is passed to the XLA runtime),
the buffer is actually donated, and the update is executed in place, as
requested by the compile-time aliasing API.

If, however, the buffer that is aliased at compile time is *not* donated at
runtime, *copy-protection* kicks in: an extra output buffer `O` is allocated,
and the contents of the input buffer `P` that was meant to be aliased are copied
into `O` (so effectively the program can execute as if the buffer `O` was
donated at runtime).
