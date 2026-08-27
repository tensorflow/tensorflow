# Metal PluggableDevice backend

GPU support for Apple silicon Macs, built into TensorFlow rather than
installed as a separate plugin.

## Status

Experimental, and deliberately narrow. This is the device and memory
foundation plus a small set of kernels: enough to place tensors on the GPU,
move data to and from it, and run an end-to-end computation. It is not yet
enough to train a model. See [Limitations](#limitations).

## Building

```
./configure    # answer yes to "Metal GPU"
bazel build --config=metal //tensorflow/tools/pip_package:wheel
```

`--config=metal` implies `--config=macos_arm64` and sets
`--define=with_metal_support=true`. Without it nothing here is compiled and no
dependency on it is added, on any platform.

Check it worked:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Set `TF_DISABLE_METAL=1` to keep the backend out of the process without
rebuilding.

## Why this exists

Apple's `tensorflow-metal` plugin was the only GPU path for TensorFlow on Mac.
Its last release, 1.2.0, is from January 2025, and it publishes wheels for
CPython 3.9 through 3.12 only, so there is no Metal plugin at all for the
Python 3.13 that current TensorFlow supports.

## Design

The backend is written as a StreamExecutor C API plugin, the same interface an
out-of-tree plugin implements, and is registered in-process rather than
`dlopen`'ed: `metal_plugin_registrar.cc` hands a `PluggableDeviceInit_Api` to
`RegisterPluggableDevicePlugin`. That reuses the whole of
`tensorflow/core/common_runtime/pluggable_device` unchanged, and keeps the code
extractable to a separate repository if it ever needs to move.

### Device type

Devices are registered under device type `GPU` with platform name `METAL`, so
they appear as `/device:GPU:0`. Existing user code, Keras, `tf.distribute` and
the placement rules therefore work without modification. There is no clash with
the CUDA GPU device because CUDA is never built on macOS.

### Unified memory is the load-bearing decision

Every allocation is `MTLResourceStorageModeShared`, and what core receives as
the device address is the buffer's `contents` pointer: a real, host-addressable
address in the same physical memory the GPU reads.

This is what makes the backend work at all. Core does not treat a device
address as opaque. The BFC allocator carves sub-allocations out of a region by
pointer arithmetic, and kernels receive pointers into the middle of buffers. A
plugin returning an `id<MTLBuffer>` in that field breaks the first time core
adds an offset to it.

It also removes the transfer cost that dominates small-model performance on
Mac. Host/device copies are `memcpy`, not staged blits.

`MetalBufferRegistry` goes the other way, recovering the `(buffer, offset)`
pair a Metal encoder needs from an arbitrary interior pointer.

The backend accepts only devices reporting `hasUnifiedMemory`, and that is a
correctness gate. On a discrete GPU, shared-storage buffers are host-side
staging that needs explicit `didModifyRange:` and `synchronizeResource:`, so
the zero-copy transfers would read stale data. Devices without it are skipped
and logged by name.

### Stream ordering

StreamExecutor's stream is strictly ordered. An `MTLCommandQueue` only
guarantees that command buffers are *scheduled* in commit order; their
execution may overlap. `SP_Stream_st` restores the contract with a per-stream
`MTLSharedEvent` used as a sequence counter: command buffer N waits for N-1 and
signals N. That serialises a stream without serialising the device.

`OrderedCommandBuffer` is the only way to obtain a command buffer, so nothing
can bypass the ordering. Its `CommitWithHostCompletion` variant signals the
sequence from the host after a completion block runs, which is what lets the
`memcpy`-based transfers take part in stream order without racing the next
command buffer.

### Kernels

Registered through the Kernel C API. Elementwise arithmetic and `Cast` are
Metal compute shaders compiled at runtime from an embedded source string;
`MatMul` uses `MPSMatrixMultiplication`.

`MPSMatrix` rather than `MPSGraph` because `MPSMatrix`'s
`initWithBuffer:offset:descriptor:` takes a byte offset, whereas
`MPSGraphTensorData`'s `MTLBuffer` initialiser assumes the tensor starts at the
beginning of the buffer. Given BFC sub-allocation, an `MPSGraph` path would
have to copy every operand into a buffer of its own first.

## Supported ops

| Op | dtypes |
| --- | --- |
| `Add`, `AddV2`, `Sub`, `Mul` | float32, float16 |
| `MatMul` | float32, float16 |
| `Cast` | float32 <-> float16 |
| `Identity` | float32, float16, int32, int64, bool |

## Limitations

* **Op coverage is minimal.** No convolutions, pooling, normalisation,
  reductions, activations or gradients, so no model trains on this yet.
  Unsupported ops fall back to the host.
* **Broadcasting** is limited to a scalar operand. Other shapes are rejected
  with both shapes named rather than computed incorrectly.
* **`MatMul`** takes rank-2 tensors only, and float16 requires an even column
  count, since an odd one produces a row stride MPS will not accept.
* **No graph optimizer or profiler module.** The `TF_InitGraph` and
  `TF_InitProfiler` hooks are unimplemented, so there is no op fusion and no
  Metal timeline in the TensorFlow profiler.
* **Single device.** Every Apple silicon Mac reports one GPU; multi-device has
  had no testing.
* **`memset32`** with a pattern that is not four equal bytes runs on the host.
  A compute shader would be faster for large buffers.

## Files

| File | Role |
| --- | --- |
| `metal_buffer_registry.{h,mm}` | Device address to `MTLBuffer` mapping, allocator stats |
| `metal_stream.{h,mm}` | Streams, events, timers, `OrderedCommandBuffer`, per-device state |
| `metal_stream_executor.{h,mm}` | The `SP_StreamExecutor` callback table |
| `metal_platform.{h,mm}` | `SP_Platform`, device discovery, plugin entry point |
| `metal_plugin_registrar.cc` | Static registration with core |
| `kernels/` | Op kernels and the shader library |
