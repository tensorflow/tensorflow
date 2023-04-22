# XLA Custom Calls

This document describes how to write and use XLA "custom calls". Custom calls
let you invoke code written in a programming language like C++ or CUDA from an
XLA program.

Warning: Custom calls are a low-level power-user feature. It is easy to break
your program in difficult-to-debug (and even difficult-to-notice) ways using
custom-calls. You shouldn't use custom calls unless you're prepared to debug XLA
yourself when something goes wrong, and you should expect relatively less
assistance from XLA developers if you run into trouble.

Warning: The custom-call API/ABI is not currently stable. We don't intend to
change it capriciously, but it may change. Some possible future changes are
described below.

## Custom-call on CPU

You can create an HLO instruction which represents a custom-call via XLA's
client API. This is not exposed via TensorFlow as of writing.

For example, the following code uses a custom-call to compute
`A[i] = B[i % 128] + C[i]` on the CPU. (Of course you could -- and should! -- do
this with regular HLO.)

```c++
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(0, xla::ShapeUtil::CreateShape(F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(1, xla::ShapeUtil::CreateShape(F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                      /*output_shape=*/ShapeUtil::CreateShape(F32, {2048}));
}

void do_custom_call(void* out, const void** in) {
  float* out_buf = reinterpret_cast<float*>(out);
  const float* in0 = reinterpret_cast<const float*>(in[0]);
  const float* in1 = reinterpret_cast<const float*>(in[1]);
  for (int i = 0; i < 2048; ++i) {
    out_buf[i] = in0[i % 128] + in1[i];
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "Host");
```

Notice that the function `do_custom_call` needs to know the dimensions of the
buffers it operates over. In this example we hardcode the sizes 128 and 2048. If
you don't want to do this, you can pass the dimensions in as parameters to the
call.

## Custom-call on GPU

The GPU custom call framework is somewhat different than that on the CPU. Here
is a CUDA example that does the same `A[i] = B[i % 128] + C[i]` computation as
the CPU code above.

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = threadIdx.x * blockSize.x + gridIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64 block_dim = 64;
  const int64 grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
```

Notice first that the GPU custom call function *is still a function executed on
the CPU*. Our `do_custom_call` CPU function is responsible for enqueueing work
on the GPU. Here it launches a CUDA kernel, but it could also do something else,
like call cublas.

`buffers` is an array of pointers which lives on the host, and each element it
contains points to device (i.e. GPU) memory. The parameters come first, followed
by the output value. This is notably different from the CPU calling convention,
which has two params, `ins` and `out`. The main reason we diverge is to make it
possible to handle tuple-shaped inputs/outputs efficiently; see the section
below.

As in the CPU example, we've hardcoded the input and output buffer sizes into
our custom call. However unlike in the CPU case, passing the buffer sizes in as
operands to the custom call would not work well. Usually we need the buffer
sizes available to us on the CPU; e.g. when launching a kernel, we need to know
the block/grid dimensions to use. But if we were to pass the buffer sizes as
operands to our custom call, their values would live in GPU memory. We'd then
have to do an expensive synchronous device-to-host memcpy at the start of our
operation just to read the sizes.

To let you work around this, we provide the `opaque` parameter. You can set this
to an arbitrary string of bytes when you create the custom call:

```c++
std::string opaque = "...";
xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
                /*output_shape=*/ShapeUtil::CreateShape(F32, {2048}),
                opaque);
```

Since `xla::Shape` has a protocol buffer representation, you could store this
serialized proto inside of `opaque` and deserialize it within your GPU
custom-call. Note however that although `xla::ShapeProto` does not change
frequently, it *does* change. Check the git log to see how it has changed in the
past.

## Passing tuples to custom-calls

Consider the following custom-call.

```c++
using xla::ShapeUtil;
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
xla::CustomCall(&b, "do_custom_call", /*operands=*/{p0}, out_shape);
```

On both CPU and GPU, a tuple is represented in memory as an array of pointers.
In C++-pseudocode, parameter 0 above is laid out as follows.

```c++
// In-memory layout of parameter 0 from custom-call above.  True on both CPU
// and GPU.
float* subbuf0 = new float[32];
float* subbuf1 = new float[64];
float* subbuf2 = new float[128]
float* subbuf3 = new float[256];

void* subtuple = new void*[2];
(*subtuple)[0] = subbuf1;
(*subtuple)[1] = subbuf2;

void* p0 = new void*[3];
(*p0)[0] = subbuf0;
(*p0)[1] = subtuple;
(*p0)[2] = subbuf3;
```

Although the in-memory representation of tuples is the same in CPU and GPU, they
are handled differently in the CPU and GPU custom-call calling conventions.

### Tuple outputs as temp buffers

Tuple inputs to custom-calls are a convenience, but they aren't strictly
necessary. If we didn't support tuple inputs to custom calls, you could always
unpack the tuples using get-tuple-element before passing them to the custom
call.

On the other hand, tuple *outputs* do let you do things you couldn't otherwise.

The obvious reason to have tuple outputs is, that's how a custom call (or any
other XLA op) returns multiple independent arrays.

But less obviously, a tuple output is also a way to give your custom call temp
memory. Yes, an *output* can represent a temp buffer. Consider, an output buffer
has the property that the op can write to it, and it can read from it after it's
been written to. That's exactly what you want from a temp buffer.

In the example above, suppose we wanted to use the `F32[1024]` as a temp buffer.
Then we'd write the HLO just as above, and we'd simply never read tuple index 1
of the custom call's output.

### Tuples in CPU custom-calls

In CPU code, we have a function `do_custom_call(const void** ins, void* out)`.
`ins` is an array with just one element, which points to `param0`. The
subbuffers of `param0` are accessible by dereferencing that pointer, and the
subbuffers of `output_tuple` are accessible by dereferencing `out`.

### Tuples in GPU custom-calls

In GPU code, we have a function `do_custom_call(..., void** buffers, ...)`. In
this case `buffers` is a host array of *six* device pointers, one for each leaf
buffer in the input/output. To generate the flat list, we iterate over the
parameters and output, and for each we do a preorder traversal of its shape.
Concretely:

```c++
// Layout of `buffers` parameter to GPU custom call function for custom-call
// above.
buffers[0] == subbuf0
buffers[1] == subbuf1
buffers[2] == subbuf2
buffers[3] == subbuf3
buffers[4] == output_subbuf0
buffers[5] == output_subbuf1
```
