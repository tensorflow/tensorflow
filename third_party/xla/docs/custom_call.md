# XLA Custom Calls

This document describes how to write and use XLA custom calls using XLA FFI
library. Custom call is a mechanism to describe an external "operation" in the
HLO module to the XLA compiler (at compile time), and XLA FFI is a mechanism to
register implementation of such operations with XLA (at run time). FFI stands
for "foreign function interface" and it is a set of C APIs that define a binary
interface (ABI) for XLA to call into external code written in other programming
languages. XLA provides header-only bindings for XLA FFI written in C++, which
hides all the low level details of underlying C APIs from the end user.

> **Caution:** The custom-call API/ABI uses PJRT-style versioning (major, minor),
> however at this point it is still experimental and can be broken at any time.
> Once API/ABI is finalized we intend to provide stability guarantees
> similar to PJRT.

> **Caution** The HLO-visible names of functions registered with the custom-call
> macros API do not respect C++ namespaces. As a result, accidental collisions
> from functions registered by different libraries are entirely possible! The
> API will reject such duplicate registrations, but to avoid issues in large
> projects the safest option is to either fully namespace-qualify all references
> to the functions in both the `XLA_REGISTER_CUSTOM_CALL` registration macros
> and custom call target references or to use C-style namespacing directly in
> the function name.

## JAX + XLA Custom Calls

See [JAX documentation](https://jax.readthedocs.io/en/latest/ffi.html) for
end to end examples of integrating custom calls and XLA FFI with JAX.

## XLA FFI Binding

XLA FFI binding is a compile-time specification of the custom call signature:
custom call arguments, attributes and their types, and additional parameters
passed via the execution context (i.e., gpu stream for GPU backend). XLA FFI
finding can be bound to any C++ callable (function pointer, lambda, etc.) with
compatible `operator()` signature. Constructed handler decodes XLA FFI call
frame (defined by the stable C API), type check all parameters, and forward
decoded results to the user-defined callback.

XLA FFI binding heavily relies on template metaprogramming to be be able to
compile constructed handler to the most efficient machine code. Run time
overheads are in order of a couple of nanoseconds for each custom call
parameter.

XLA FFI customization points implemented as template specializations, and
users can define how to decode their custom types, i.e., it is possible
to define custom decoding for user-defined `enum class` types.

### Returning Errors From Custom Calls

Custom call implementations must return `xla::ffi::Error` value to signal
success or error to XLA runtime. It is similar to `absl::Status`, and has
the same set of error codes. We do not use `absl::Status` because it does
not have a stable ABI and it would be unsafe to pass it between dynamically
loaded custom call library, and XLA itself.

```c++
// Handler that always returns an error.
auto always_error = Ffi::Bind().To(
    []() { return Error(ErrorCode::kInternal, "Oops!"); });

// Handler that always returns a success.
auto always_success = Ffi::Bind().To(
    []() { return Error::Success(); });

```

### Buffer Arguments And Results

XLA uses destination passing style for results: custom calls (or any other XLA
operations for that matter) do not allocate memory for results, and instead
write into destinations passed by XLA runtime. XLA uses static buffer
assignment, and allocates buffers for all values based on their live ranges at
compile time.

Results passed to FFI handlers wrapped into a `Result<T>` template, that
has a pointer-like semantics: `operator->` gives access to the underlying
parameter.

`AnyBuffer` arguments and results gives access to custom call buffer parameters
of any data type. This is useful when custom call has a generic implementation
that works for multiple data types, and custom call implementation does run time
dispatching based on data type. `AnyBuffer` gives access to the buffer data
type, dimensions, and a pointer to the buffer itself.

```mlir
%0 = "stablehlo.custom_call"(%arg0) {
  call_target_name = "foo",
  api_version = 4 : i32
} : (tensor<2x2xf32>) -> tensor<2x2xf32>
```


```c++
// Buffers of any rank and data type.
auto handler = Ffi::Bind().Arg<AnyBuffer>().Ret<AnyBuffer>().To(
    [](AnyBuffer arg, Result<AnyBuffer> res) -> Error {
      void* arg_data = arg.untyped_data();
      void* res_data = res->untyped_data();
      return Error::Success();
    });
```

### Constrained Buffer Arguments And Results

`Buffer` allows to add constraints on the buffer data type and rank, and they
will be automatically checked by the handler and return an error to XLA runtime,
if run time arguments do not match the FFI handler signature.

```c++
// Buffers of any rank and F32 data type.
auto handler = Ffi::Bind().Arg<Buffer<F32>>().Ret<Buffer<F32>>().To(
    [](Buffer<F32> arg, Result<Buffer<F32>> res) -> Error {
      float* arg_data = arg.typed_data();
      float* res_data = res->typed_data();
      return Error::Success();
    });
```

```c++
// Buffers of rank 2 and F32 data type.
auto handler = Ffi::Bind().Arg<BufferR2<F32>>().Ret<BufferR2<F32>>().To(
    [](BufferR2<F32> arg, Result<BufferR2<F32>> res) -> Error {
      float* arg_data = arg.typed_data();
      float* res_data = res->typed_data();
      return Error::Success();
    });
```

### Variadic Arguments And Results

If the number of arguments and result can be different in different instances of
a custom call, they can be decoded at run time using `RemainingArgs` and
`RemainingRets`.

```
auto handler = Ffi::Bind().RemainingArgs().RemainingRets().To(
    [](RemainingArgs args, RemainingRets results) -> Error {
      ErrorOr<AnyBuffer> arg = args.get<AnyBuffer>(0);
      ErrorOr<Result<AnyBuffer>> res = results.get<AnyBuffer>(0);

      if (!arg.has_value()) {
        return Error(ErrorCode::kInternal, arg.error());
      }

      if (!res.has_value()) {
        return Error(ErrorCode::kInternal, res.error());
      }

      return Error::Success();
    });
```

Variadic arguments and results can be declared after regular arguments and
results, however binding regular arguments and results after variadic one is
illegal.

```c++
auto handler =
    Ffi::Bind()
        .Arg<AnyBuffer>()
        .RemainingArgs()
        .Ret<AnyBuffer>()
        .RemainingRets()
        .To([](AnyBuffer arg, RemainingArgs args, AnyBuffer ret,
               RemainingRets results) -> Error { return Error::Success(); });
```

### Attributes

XLA FFI supports automatic decoding of `mlir::DictionaryAttr` passed as a
`custom_call` `backend_config` into FFI handler arguments.

Note: See [stablehlo RFC](https://github.com/openxla/stablehlo/blob/main/rfcs/20240312-standardize-customcallop.md)
for details, and `stablehlo.custom_call` operation specification.

```mlir
%0 = "stablehlo.custom_call"(%arg0) {
  call_target_name = "foo",
  backend_config= {
    i32 = 42 : i32,
    str = "string"
  },
  api_version = 4 : i32
} : (tensor<f32>) -> tensor<f32>
```

In this example custom call has a single buffer argument and two attributes, and
XLA FFI can automatically decode them and pass to the user-defined callable.

```c++
auto handler = Ffi::Bind()
  .Arg<BufferR0<F32>>()
  .Attr<int32_t>("i32")
  .Attr<std::string_view>("str")
  .To([](BufferR0<F32> buffer, int32_t i32, std::string_view str) {
    return Error::Success();
  });
```

### User-Defined Enum Attributes

XLA FFI can automatically decode integral MLIR attributes into user-defined
enums. Enum class must have the same underlying integral type, and decoding
has to be explicitly registered with XLA FFI.


```mlir
%0 = "stablehlo.custom_call"(%arg0) {
  call_target_name = "foo",
  backend_config= {
    command = 0 : i32
  },
  api_version = 4 : i32
} : (tensor<f32>) -> tensor<f32>
```

```c++
enum class Command : int32_t {
  kAdd = 0,
  kMul = 1,
};

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(Command);

auto handler = Ffi::Bind().Attr<Command>("command").To(
    [](Command command) -> Error { return Error::Success(); });
```

### Binding All Custom Call Attributes

It is possible to get access to all custom call attributes as a dictionary
and lazily decode only the attributes that are needed at run time.

```c++
auto handler = Ffi::Bind().Attrs().To([](Dictionary attrs) -> Error {
  ErrorOr<int32_t> i32 = attrs.get<int32_t>("i32");
  return Error::Success();
});
```

### User-defined Struct Attributes

XLA FFI can decode dictionary attributes into user-defined structs.

```mlir
%0 = "stablehlo.custom_call"(%arg0) {
  call_target_name = "foo",
  backend_config= {
    range = { lo = 0 : i64, hi = 42 : i64 }
  },
  api_version = 4 : i32
} : (tensor<f32>) -> tensor<f32>
```

In example above `range` is an `mlir::DictionaryAttr` attribute, and instead
of accessing dictionary fields by name, it can be automatically decoded as
a C++ struct. Decoding has to be explicitly registered with a
`XLA_FFI_REGISTER_STRUCT_ATTR_DECODING` macro (behind the scene it defines
a template specialization in `::xla::ffi` namespace, thus macro must be added to
the global namespace).

```c++
struct Range {
  int64_t lo;
  int64_t hi;
};

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(Range, StructMember<int64_t>("lo"),
                                             StructMember<int64_t>("hi"));

auto handler = Ffi::Bind().Attr<Range>("range").To([](Range range) -> Error{
  return Error::Success();
});
```

Custom attributes can be loaded from a dictionary, just like any other
attribute. In example below, all custom call attributes decoded as a
`Dictionary`, and a `range` can be accessed by name.

```c++
auto handler = Ffi::Bind().Attrs().To([](Dictionary attrs) -> Error {
  ErrorOr<Range> range = attrs.get<Range>("range");
  return Error::Success();
});
```

## Create a custom call on CPU

You can create an HLO instruction that represents a custom call via XLA's client
API. For example, the following code uses a custom call to compute `A[i] = B[i %
128]+ C[i]` on the CPU. (Of course you could &ndash; and should! &ndash; do this
with regular HLO.)

```c++
#include "xla/client/xla_builder.h"
#include "xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla::F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
        /*shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
        /*opaque=*/"", /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
        /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
}

// Constrain custom call arguments to rank-1 buffers of F32 data type.
using BufferF32 = xla::ffi::BufferR1<xla::ffi::DataType::F32>;

// Implement a custom call as a C+ function. Note that we can use `Buffer` type
// defined by XLA FFI that gives us access to buffer data type and shape.
xla::ffi::Error do_custom_call(BufferF32 in0, BufferF32 in1,
                               xla::ffi::Result<BufferF32> out) {
  size_t d0 = in0.dimensions[0];
  size_t d1 = in1.dimensions[0];

  // Check that dimensions are compatible.
  assert(out->dimensions[0] == d1 && "unexpected dimensions");

  for (size_t i = 0; i < d1; ++i) {
    out->data[i] = in0.data[i % d0] + in1.data[i];
  }
}

// Explicitly define an XLA FFI handler signature and bind it to the
// `do_custom_call` implementation. XLA FFI handler can automatically infer
// type signature from the custom call function, but it relies on magical
// template metaprogramming an explicit binding provides and extra level of
// type checking and clearly states custom call author intentions.
XLA_FFI_DEFINE_HANDLER(handler, do_custom_call,
                       ffi::Ffi::Bind()
                           .Arg<Buffer>()
                           .Arg<Buffer>()
                           .Ret<Buffer>());

// Registers `handler` with and XLA FFI on a "Host" platform.
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "do_custom_call",
                         "Host", handler);
```

## Create a custom call on GPU

The GPU custom call registration with XLA FFI is almost identical, the only
difference is that for GPU you need to ask for an underlying platform stream
(CUDA or ROCM stream) to be able to launch kernel on device. Here is a CUDA
example that does the same computation (`A[i] = B[i % 128] + C[i]`) as the CPU
code above.

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, BufferF32 in0, BufferF32 in1,
                    xla::ffi::Result<BufferF32> out) {
  size_t d0 = in0.dimensions[0];
  size_t d1 = in1.dimensions[0];
  size_t d2 = out->dimensions[0];

  assert(d0 == 128 && d1 == 2048 && d2 == 2048 && "unexpected dimensions");

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim, 0, stream>>>(
    in0.data, in1.data, out->data);
}

XLA_FFI_DEFINE_HANDLER(handler, do_custom_call,
                       ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<CUstream>>()
                           .Arg<BufferF32>()
                           .Arg<BufferF32>()
                           .Ret<BufferF32>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "do_custom_call",
                         "CUDA", handler);
```

Notice first that the GPU custom call function *is still a function executed on
the CPU*. The `do_custom_call` CPU function is responsible for enqueueing work
on the GPU. Here it launches a CUDA kernel, but it could also do something else,
like call cuBLAS.

Arguments and results also live on the host, and data member contains a pointer
to device (i.e. GPU) memory. Buffers passed to custom call handler have the
shape of the underlying device buffers, so the custom call can compute kernel
launch parameters from them.

## Passing tuples to custom calls

Consider the following custom call.

```c++
using xla::ShapeUtil;
using xla::F32;
Shape p0_shape = ShapeUtil::MakeTuple({
    ShapeUtil::MakeShape(F32, {32}),
    ShapeUtil::MakeTuple({
        ShapeUtil::MakeShape(F32, {64}),
        ShapeUtil::MakeShape(F32, {128}),
    }),
    ShapeUtil::MakeShape(F32, {256}),
});
xla::XlaOp p0 = xla::Parameter(0, p0_shape, "p0");

Shape out_shape = ShapeUtil::MakeTuple({
  ShapeUtil::MakeShape(F32, {512}),
  ShapeUtil::MakeShape(F32, {1024}),
});
xla::CustomCall(&b, "do_custom_call", /*operands=*/{p0}, out_shape, ...);
```

On both CPU and GPU, a tuple is represented in memory as an array of pointers.
When XLA calls custom calls with tuple arguments or results it flattens them and
passes as regular buffer arguments or results.

### Tuple outputs as temp buffers

Tuple inputs to custom calls are a convenience, but they aren't strictly
necessary. If we didn't support tuple inputs to custom calls, you could always
unpack the tuples using get-tuple-element before passing them to the custom
call.

On the other hand, tuple *outputs* do let you do things you couldn't otherwise.

The obvious reason to have tuple outputs is that tuple outputs are how a custom
call (or any other XLA op) returns multiple independent arrays.

But less obviously, a tuple output is also a way to give your custom call temp
memory. Yes, an *output* can represent a temp buffer. Consider, an output buffer
has the property that the op can write to it, and it can read from it after it's
been written to. That's exactly what you want from a temp buffer.

In the example above, suppose we wanted to use the `F32[1024]` as a temp buffer.
Then we'd write the HLO just as above, and we'd simply never read tuple index 1
of the custom call's output.
