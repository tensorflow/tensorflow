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

## Late Binding

In some cases, the operands (inputs) or outputs of an asynchronous
operation are not all available or allocated when the operation starts.
XLA supports *late binding*, which allows operands to be incrementally
bound during `async-update` steps, and outputs to be bound during either
`async-update` or `async-done` steps.

### Representation in HLO

For a called computation that expects $N$ parameters, we can start the
asynchronous execution with fewer than $N$ operands. The remaining
operands are passed in subsequent `async-update` instructions.

*   `async-start` binds the first $K$ operands ($K < N$).
*   `async-update` instructions bind the remaining $N - K$ operands.

Operand bindings must happen in left-to-right order. That is, if a
computation expects parameters $P_0, P_1, \dots, P_{N-1}$, they must be bound in
that order across the async chain.

The `async-start` and `async-update` shapes reflect the incrementally
bound parameters. Specifically, the first element of the tuple shape
(the operand shapes) grows as more operands are bound.

Output binding is independent of operand binding and can happen at any
step in the async chain (either in an `async-update` or at the final
`async-done`).

### Example with `kCall`

Consider a called computation `%foo` that takes two parameters:

```
%foo {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}
```

We can call this computation asynchronously, binding `%p0` at start and
`%p1` at update:

```
%call-start = ((f32[]), (), s32[]) call-start(%operand0), to_apply=%foo
%call-update = ((f32[], f32[]), f32[], s32[]) call-update(%call-start, %operand1)
%result = f32[] call-done(%call-update)
```

The parser desugars this into the following HLO:

```
%async-start = ((f32[]), (), s32[]) async-start(%operand0), calls=%foo
%async-update = ((f32[], f32[]), f32[], s32[]) async-update(%async-start, %operand1)
%result = f32[] async-done(%async-update)
```

### Late-Bound Outputs

In addition to operands (inputs), the **outputs** of an asynchronous
operation can also be bound late. This is useful when the output
buffers are not known or allocated at the start of the operation.

To represent late-bound outputs:
1.  The `async-start` (or `call-start`) instruction is defined with an
    empty tuple `()` at index 1 of its output shape (the result slot).
2.  A subsequent `async-update` (or `call-update`) instruction
    specifies the actual output shape at index 1, replacing the empty
    tuple.
3.  Alternatively, the output can be bound at the end of the chain by
    the `async-done` (or `call-done`) instruction, which returns the
    final output shape. This can be done regardless of whether there are
    intermediate `async-update` steps in the chain.

#### Example with `async-update`

```
// Output is not bound at start (index 1 is ())
%call-start = ((f32[1024]), (), s32[]) call-start(%input_buffer), to_apply=%foo

// Output is bound at update (index 1 becomes (f32[1024]))
%call-update = ((f32[1024]), (f32[1024]), s32[]) call-update(%call-start, %output_buffer)

%result = (f32[1024]) call-done(%call-update)
```

The parser desugars this into:

```
%async-start = ((f32[1024]), (), s32[]) async-start(%input_buffer), calls=%foo
%async-update = ((f32[1024]), (f32[1024]), s32[]) async-update(%async-start, %output_buffer)
%result = (f32[1024]) async-done(%async-update)
```

#### Example with `async-done` (without `async-update`)

If there are no intermediate update steps, the output can be bound
directly at `async-done`:

```
// Output is not bound at start (index 1 is ())
%call-start = ((f32[1024]), (), s32[]) call-start(%input_buffer), to_apply=%foo

// Output is bound at done
%result = (f32[1024]) call-done(%call-start)
```

The parser desugars this into:

```
%async-start = ((f32[1024]), (), s32[]) async-start(%input_buffer), calls=%foo
%result = (f32[1024]) async-done(%async-start)
```

#### Example with intermediate `async-update` and output bound at `async-done`

If there are intermediate update steps to bind operands, but the output
is still bound at the very end:

```
// Output is not bound at start, no operands bound
%call-start = ((), (), s32[]) call-start(), to_apply=%foo

// Operands are bound at update, but output remains unbound (index 1 is ())
%call-update = ((f32[], f32[]), (), s32[]) call-update(%call-start, %operand0, %operand1)

// Output is bound at done
%result = f32[] call-done(%call-update)
```

The parser desugars this into:

```
%async-start = ((), (), s32[]) async-start(), calls=%foo
%async-update = ((f32[], f32[]), (), s32[]) async-update(%async-start, %operand0, %operand1)
%result = f32[] async-done(%async-update)
```
