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

%async-start = (f32[64], f32[32], s32[]) async-start(f32[64] %operand),
                                         calls=%async_op
%async-done = f32[32] async-done((f32[64], f32[32], s32[]) %async-start)
```

In the representation above, only `async-start` has a called computation since
it is trivial to find what the `async-done` does by following its operand to
find the corresponding `async-start` to find the called computation.

Also note
that the first element in the output tuple of `async-start` aliases with the
operand, so the buffer stays alive until at least the async-done instruction.
Similarly, the second element aliases with the output of `async-done`, and the
third element is the context state that is used to keep track of the
asynchronous operation. This representation also supports multiple tensors in
the asynchronous operation input and/or output and the aliasing works the same
way:

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

%async-start = (f32[64], f32[32], s32[]) async-start(f32[64] %operand),
                                         calls=%async_op
%async-update0 = (f32[64], f32[32], s32[]) async-update(
                           (f32[64], f32[32], s32[]) %async-start)
%async-update1 = (f32[64], f32[32], s32[]) async-update(
                           (f32[64], f32[32], s32[]) %async-update0)
%async-done = f32[32] async-done((f32[64], f32[32], s32[]) %async-update1)

```

## Syntax sugar

Since having a separate computation to define the operation that will be
performed asynchronously is a bit cumbersome, we also propose a syntax sugar to
automatically print and parse asynchronous operations as if they are first-class
opcodes. The idea is to treat the “-start”,  “-update”, and “-done” suffixes
specially by automatically creating the computation and instruction (without the
suffix) when parsing. For example, the code snippet above can be pretty-printed
to the following and the two can be parsed to the same representation:

```
%op-start = (f32[64], f32[32], s32[]) op-start(f32[64] %operand),
                                      op_specific_attr=”foo”
%op-update0 = (f32[64], f32[32], s32[]) op-update(
                        (f32[64], f32[32], s32[]) %op-start),
                        op_specific_attr=”foo”
%op-update1 = (f32[64], f32[32], s32[]) op-update(
                        (f32[64], f32[32], s32[]) %op-update0)
%op-done = f32[32] op-done((f32[64], f32[32], s32[]) %op-update1)

```

In order not to create ambiguities, the verifier will not allow an operation to
be wrapped with async-start if we explicitly defined an opcode for that
operation with the “-start” and/or “-done” suffixes. This is also an escape
hatch in case we have any instructions that require HLO-level treatment that
doesn’t fit in the model described above (e.g. the aliasing input/output
buffers). So, initially, `copy-start`/`copy-done`,
`collective-permute-start`/`collective-permute-done` etc. will continue to use
their respective first-class opcodes instead of the new
`async-start`/`async-done` opcodes until we clean up the code to remove these
“-start”/”-done” opcodes.
