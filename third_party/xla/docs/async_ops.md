# Async HLO Instructions

1. Adding async operations to HLO is cumbersome (i.e. `all-reduce-start` and
   `all-reduce-done`).
2. The start and done split may be inadequate for some of the asynchronous use
   cases.

To target the first shortcoming, we propose to introduce one last set of new
asynchronous opcodes: `kAsyncStart`, `kAsyncUpdate`, and `kAsyncDone`. The idea
is to create a generic asynchronous opcode that can wrap any HLO instruction.
The actual operation that will be performed asynchronously will be encoded using
a called computation that only has the instruction as its root and any
parameters for inputs. The in-flight input/output buffer handling and aliasing
can then be shared for any asynchronous operation. The async-start instruction’s
output shape will then be a tuple of the input operands, output values, and any
intermediate state that is needed for the `async-update` or `async-done`
instructions.

```
%async_op {
  %param0 = f32[64] parameter(0)
  ROOT %op = f32[32] op(f32[64] %param0), op_specific_attr=”foo”
}

%async-start = ((f32[64]), f32[32], s32[]) async-start(f32[64] %operand),
                                           calls=%async_op
%async-done = f32[32] async-done(((f32[64]), f32[32], s32[]) %async-start)
```

In the representation above, only `async-start` has a called computation since
it is trivial to find what the `async-done` does by following its operand to
find the corresponding `async-start` to find the called computation.

Also note that the first element in the output tuple of `async-start` is a
tuple containing the operands. The elements of this operand tuple alias with
the respective operands, so their buffers stay alive until at least the
`async-done` instruction. Similarly, the second element aliases with the output
of `async-done`, and the third element is the context state that is used to
keep track of the asynchronous operation. This representation naturally
supports multiple tensors in the asynchronous operation input and/or output:

```
%async_op {
  %param0 = f32[64] parameter(0)
  %param1 = f32[64] parameter(1)
  ROOT %op = (f32[32], f32[32]) op(f32[64] %param0, f32[64] %param1),
                                op_specific_attr=”foo”
}

%async-start = ((f32[64], f32[64]), (f32[32], f32[32]), s32[])
               async-start(f32[64] %operand0, f32[64] %operand1),
               calls=%async_op
%async-done = (f32[32], f32[32]) async-done(%async-start)
```

In addition, the op can further be decomposed into zero or more `async-update`
steps that perform intermediate computations. The input/output aliasing works
the same way with the `async-update` instruction and each `async-start` and
`async-update` instructions must have one user that is either another
`async-update` or an `async-done`:

```
%async_op {
  %param0 = f32[64] parameter(0)
  ROOT %op = f32[32] op(f32[64] %param0), op_specific_attr=”foo”
}

%async-start = ((f32[64]), f32[32], s32[]) async-start(f32[64] %operand),
                                         calls=%async_op
%async-update0 = ((f32[64]), f32[32], s32[]) async-update(
                           ((f32[64]), f32[32], s32[]) %async-start)
%async-update1 = ((f32[64]), f32[32], s32[]) async-update(
                           ((f32[64]), f32[32], s32[]) %async-update0)
%async-done = f32[32] async-done(((f32[64]), f32[32], s32[]) %async-update1)

```

## Syntax sugar

The HLO parser supports syntax sugar to automatically parse and print
asynchronous operations as if they are first-class opcodes. The parser treats
the `-start`, `-update`, and `-done` suffixes specially by automatically
creating the async computation and the wrapped instruction (without the suffix).

For example, an asynchronous `custom-call` can be written as:

```
%cc-start = ((f32[64]), f32[32], s32[]) custom-call-start(%operand),
                                        custom_call_target="foo"
%cc-update = ((f32[64]), f32[32], s32[]) custom-call-update(%cc-start)
%result = f32[32] custom-call-done(%cc-update)
```

The parser desugars this into the following equivalent HLO:

```
%async_computation {
  %p0 = f32[64] parameter(0)
  ROOT %custom-call = f32[32] custom-call(%p0), custom_call_target="foo"
}

%async-start = ((f32[64]), f32[32], s32[]) async-start(%operand),
                calls=%async_computation
%async-update = ((f32[64]), f32[32], s32[]) async-update(%async-start)
%result = f32[32] async-done(%async-update)
```

This desugaring is supported for most HLO opcodes (e.g., `custom-call`, `dot`,
`all-reduce`, etc.).

### Exceptions

In order not to create ambiguities, the parser will not desugar operations that
have explicit first-class opcodes defined with the `-start` and/or `-done`
suffixes (e.g., `copy-start`/`copy-done`,
`collective-permute-start`/`collective-permute-done`). These will continue to
use their respective first-class opcodes.
