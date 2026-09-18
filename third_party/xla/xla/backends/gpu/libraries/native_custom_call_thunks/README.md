# Native custom-call handlers (XLA:GPU)

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

A *native custom-call handler* lowers an `HloCustomCallInstruction` directly
into a `ThunkSequence` at compile time, instead of wrapping it in a
`CustomCallThunk`.

Use it when the work behind a custom call is something XLA's runtime already
knows how to do - launching a precompiled kernel, for example. The custom call
then becomes an ordinary part of the thunk graph: it is scheduled against other
thunks, it can be captured into a command buffer, and it costs no dispatch
through the FFI boundary.

Keep using [XLA FFI](../../../ffi/README.md) when your code needs to run at
execution time with access to the stream, allocate scratch memory itself, or
live outside the XLA binary. A native handler runs inside the compiler and is
statically linked into it, so it is only an option for code that ships with
XLA or with a Google-internal XLA distribution.

## Writing a handler

A handler is a function

```c++
absl::StatusOr<ThunkSequence>(const HloCustomCallInstruction& instr,
                              const NativeCustomCallEmitterContext& ctx);
```

registered under a custom-call target name:

```c++
XLA_GPU_REGISTER_NATIVE_CUSTOM_CALL_HANDLER("my.target", MyHandler);
```

Registration happens at static-initialization time, so the library defining the
handler must be linked into the compiler and marked `alwayslink = True`.
`ThunkEmitter` consults the registry *after* all built-in specialized emitters
and *before* falling back to the FFI path, so a handler cannot shadow a
built-in lowering. Only packages on the `handler_allowlist` package group in
this package's `BUILD` file may register handlers.

`instr` is the custom call being lowered. Everything else a handler is allowed
to know about the compilation comes from `ctx`; see
`native_custom_call_emitter_context.h`.

## A worked example

The following is the demonstrator in
[`write_value_thunk_folded/`](write_value_thunk_folded/write_value_thunk_folded_handler.cc),
which lowers a custom call to a single kernel launch that fills the output
buffer with a constant read from the backend config.

```c++
absl::StatusOr<ThunkSequence> WriteValueThunkFoldedHandler(
    const HloCustomCallInstruction& instr,
    const NativeCustomCallEmitterContext& ctx) {
  ABSL_ASSIGN_OR_RETURN(ShapedSlice result, GetSingleResultShapedSlice(instr, ctx));
  int64_t num_elements = ShapeUtil::ElementsIn(result.shape);

  ABSL_ASSIGN_OR_RETURN(xla::ffi::Attributes attrs, ctx.GetFfiAttributes());
  ABSL_ASSIGN_OR_RETURN(int32_t val, attrs.Get<int32_t>("val"));

  ABSL_ASSIGN_OR_RETURN(stream_executor::KernelLoaderSpec kernel_spec,
                   stream_executor::cuda::FindCudaRuntimeKernel(
                       stream_executor::cuda::GetWriteValueFoldedKernel()));

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_args,
                   ctx.CreateKernelArguments());
  ABSL_ASSIGN_OR_RETURN(ShapeIndex result_index, SingleResultShapeIndex(instr));
  ABSL_ASSIGN_OR_RETURN(int64_t result_arg, kernel_args.ResultIndex(result_index));

  stream_executor::KernelArgsPackingSpec packing_spec;
  packing_spec.AddAddressArgument(result_arg);
  packing_spec.AddConstantArgument<int32_t>(val);

  return MakeCustomKernelThunkSequence(
      ctx,
      {/*name=*/"write_value_thunk_folded",
       /*kernel_spec=*/std::move(kernel_spec),
       /*packing_spec=*/std::move(packing_spec),
       /*block_dims=*/stream_executor::BlockDim(num_elements)},
      kernel_args);
}
```

`CustomKernelLaunchSpec` is an aggregate, but XLA still builds as C++17, so it
has to be filled in with positional list initialization rather than designated
initializers; spell the field names out in comments. Fields you do not mention
keep their defaults. To set a late field without spelling out the earlier ones,
build the spec as a named variable and assign:

```c++
CustomKernelLaunchSpec spec{/*name=*/"my_kernel",
                            /*kernel_spec=*/std::move(kernel_spec),
                            /*packing_spec=*/std::move(packing_spec)};
spec.use_pdl = true;
return MakeCustomKernelThunkSequence(ctx, std::move(spec), kernel_args);
```

## Reading attributes

`ctx.GetFfiAttributes()` returns the custom call's backend config decoded into
an `xla::ffi::Attributes`, which offers the same typed accessors an FFI handler
gets from `ffi::Dictionary`:

```c++
ABSL_ASSIGN_OR_RETURN(float epsilon, attrs.Get<float>("epsilon"));
ABSL_ASSIGN_OR_RETURN(auto dims, attrs.Get<absl::Span<const int64_t>>("dims"));
ABSL_ASSIGN_OR_RETURN(auto mode, attrs.Get<absl::string_view>("mode"));
```

Decoding is strict about widths: an attribute written as `i32` must be read as
`int32_t`, not as `int64_t`. Values that alias the attribute storage
(`absl::string_view`, `absl::Span`, nested `ffi::Dictionary`) are only valid
while the `Attributes` object is alive.

## Buffers, kernel arguments and packing specs

Three index spaces are involved in a kernel launch, and they are not the same
thing:

1.  **Operand and result positions of the HLO instruction.** What the frontend
    wrote.
2.  **Positions in `emitters::KernelArguments`.** Operands in operand order,
    then the array leaves of the result shape in shape-index order, then any
    unmanaged arguments. `CustomKernelThunk` reports buffer uses in this order,
    and `zeroed_output_buffer_indices` is expressed in it.
3.  **The kernel's parameter list.** Defined by the
    `stream_executor::KernelArgsPackingSpec`, whose relocations refer to
    positions in (2).

Get from (1) to (2) with `KernelArguments::OperandIndex` and `ResultIndex`
rather than by counting arguments yourself; that stays correct when an operand
is added or when the result shape gains a leaf.

Build (2) with `ctx.CreateKernelArguments()`. Constructing
`emitters::KernelArgument` objects by hand skips alignment, aliasing and slice
deduplication, leaves the slice index uninitialized, and makes you responsible
for the `written` flag. `written` feeds `CustomKernelThunk::buffer_uses()` and
therefore thunk scheduling, so getting it wrong produces a data race rather
than an obvious failure.

`MakeCustomKernelThunkSequence` checks (2) and (3) against each other and
against the kernel's declared arity before building the thunk.

## Resolving buffers

Always go through the context (`ctx.GetOperandShapedSlice`,
`ctx.GetResultShapedSlice`, `ctx.CreateKernelArguments`) rather than reaching
for a `BufferAssignment`. A specialized emitter may have installed an
allocation override for the instruction being lowered; the buffer assignment
does not know about those, and a handler that consults it directly would launch
its kernel on the wrong memory.

## Querying the target device

A handler runs ahead of time, so there is no device and no stream to query -
this is the main practical difference from an FFI `Instantiate` call. Use
`ctx.GetDeviceDescription()`, or `GetCudaComputeCapability(ctx)` for a
CUDA-only handler. Kernel-selection code that would normally inspect the
current device has to be refactored to take a compute capability as a
parameter.

## Testing

`NativeCustomCallHandlerTester` runs a handler over a module given as HLO text,
with a real buffer assignment and without needing a GPU:

```c++
ASSERT_OK_AND_ASSIGN(auto tester, NativeCustomCallHandlerTester::Create(R"(
  ENTRY e {
    p0 = f32[2,3] parameter(0)
    ROOT c = f32[4] custom-call(p0), custom_call_target="my.target",
      backend_config="{val = 7 : i32}"
  }
)"));
ASSERT_OK_AND_ASSIGN(ThunkSequence thunks, tester->EmitThunks());
```

Pass a different `Options::gpu_model` to test a handler's device-specific
paths; a single test binary can cover several devices without owning any of
them. `EmitThunksWith` invokes a handler directly if it is not registered in
the binary under test.

This is a compile-time test: it checks that the handler produces the thunks you
expect. Verifying that those thunks compute the right numbers still needs an
execution test on real hardware, as in
[`write_value_thunk_folded/`](write_value_thunk_folded/).

## Limitations

-   **Operand types are not validated for you.** An FFI binding such as
    `.Arg<ffi::Buffer<xla::BF16>>()` rejects a wrongly typed operand before the
    handler runs. A native handler receives the instruction as-is and must
    check element types itself if the kernel depends on them.
-   **A handler cannot decline.** Returning an error fails the compilation;
    there is no way to say "not this one, please fall back to the FFI path".
    A handler must therefore be able to lower every instance of its target.
