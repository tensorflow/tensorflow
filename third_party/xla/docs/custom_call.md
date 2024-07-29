# XLA custom calls

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
