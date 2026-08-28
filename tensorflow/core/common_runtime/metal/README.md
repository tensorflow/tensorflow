# Metal PluggableDevice backend

GPU support for Apple silicon Macs, built into TensorFlow rather than
installed as a separate plugin.

## Status

The device and memory foundation, plus enough kernels to prove the compute
path end to end. Building with `--config=metal` on an Apple silicon Mac gives
a `/device:GPU:0` that allocates, copies, synchronises, and runs the ops
listed below. Anything else falls back to the host, which is correct and slow.

This is deliberately a small first change. The op coverage that makes the
backend useful for training is a separate discussion and separate changes; the
foundation is what has to be right first, because everything else is built on
its memory model.

## Building

```
./configure    # answer yes to "Metal GPU"
bazel test --config=metal //tensorflow/core/common_runtime/metal/...
```

Without `--config=metal` the `if_metal()` select collapses to its default on
every platform: no Objective-C++ is compiled, no dependency is added, and the
Linux and Windows build graphs are unchanged. Every target in the new packages
is `target_compatible_with = ["@platforms//os:macos"]`, so wildcard builds
elsewhere skip them rather than failing.

`TF_DISABLE_METAL=1` keeps the backend out of the process without a rebuild.

## Why a PluggableDevice rather than a device factory

The backend implements the StreamExecutor C API, the same interface an
out-of-tree plugin implements, and registers it in-process instead of through
`dlopen`. `RegisterPluggableDevicePlugin` already has an overload taking a
`PluggableDeviceInit_Api` by pointer; until now only tests used it.

That reuses the whole of `tensorflow/core/common_runtime/pluggable_device`
unchanged, and keeps the backend extractable to a separate repository if that
turns out to be the right home.

The platform reports device type `GPU` and platform name `METAL`, so devices
appear as `/device:GPU:0` and existing placement, Keras and `tf.distribute`
code works without modification. CUDA is never built on macOS, so there is no
collision.

## Unified memory is the load-bearing decision

Apple silicon has one physical memory, and this is what makes the backend
worth having: host and device copies are `memcpy` rather than staged blits,
which removes the transfer cost that dominates small-model performance on a
Mac.

Only devices reporting `hasUnifiedMemory` are accepted, and that is a
correctness gate rather than a preference. On a discrete GPU, shared-storage
buffers are host-side staging that needs explicit `didModifyRange:` and
`synchronizeResource:`, so the zero-copy transfers would silently read stale
data. Skipped devices are logged by name.

## The buffer registry

TensorFlow does not treat a device address as opaque. The BFC allocator carves
sub-allocations out of a region by pointer arithmetic, and kernels receive
interior pointers. A plugin that returned an `id<MTLBuffer>` in that field
would break the first time core added an offset to it.

`MetalBufferRegistry` recovers the `(buffer, offset)` pair a Metal encoder
needs from an arbitrary interior pointer. Every kernel goes through it, and
`metal_buffer_registry_test.mm` covers the cases that matter: an interior
pointer, the boundary between two allocations, and a pointer that belongs to
no allocation at all.

## Stream ordering

A `SP_Stream` is an `MTLCommandQueue` plus an `MTLSharedEvent` and a counter.
Every command buffer waits for the previous value and signals the next, so
work executes in the order TensorFlow enqueued it even though Metal command
buffers may otherwise complete out of order.

## Supported ops

| Op | dtypes |
| --- | --- |
| `Add`, `AddV2`, `Sub`, `Mul`, `Div`, `RealDiv` | float32, float16 |
| `Maximum`, `Minimum`, `Pow`, `SquaredDifference` | float32, float16 |
| `Neg`, `Sqrt`, `Rsqrt`, `Exp`, `Log`, `Square`, `Abs`, `Reciprocal` | float32, float16 |
| `Tanh`, `Sigmoid`, `Elu`, `Selu`, `Softplus` | float32, float16 |
| `MatMul` | float32, float16 |
| `Identity` | int32 |

Binary ops broadcast with NumPy's rules. `MatMul` takes rank-2 tensors, and
float16 requires an even column count, since an odd one produces a row stride
MPS will not accept.

`Identity` is registered for int32 alone on purpose: TensorFlow registers it
for `DEVICE_GPU` itself, outside any CUDA guard, for every other number type
and for bool, so those apply to this device already. Registering them again
would give TensorFlow two registrations it cannot choose between, and it
refuses to dispatch an op whose registrations tie.

Resource variables, `Reshape`, `Const`, `Shape`, `StridedSlice` and `Pack`
need no kernel here: TensorFlow registers them for `DEVICE_DEFAULT`, which any
device type inherits when it has no kernel of its own.

## Limitations

* **Op coverage is deliberately small.** See Status.
* **No graph optimizer or profiler module.** The `TF_InitGraph` and
  `TF_InitProfiler` hooks are unimplemented, so there is no op fusion and no
  Metal timeline in the TensorFlow profiler.
* **Single device.** Every Apple silicon Mac reports one GPU; multi-device has
  had no testing.
* **`memset32`** with a pattern that is not four equal bytes runs on the host.

## Files

| File | Role |
| --- | --- |
| `metal_platform.{h,mm}` | `SP_Platform` and its function tables |
| `metal_stream_executor.{h,mm}` | the forty-odd StreamExecutor callbacks |
| `metal_stream.{h,mm}` | command queues, ordering events, timers |
| `metal_buffer_registry.{h,mm}` | interior pointer to `(buffer, offset)` |
| `metal_plugin_registrar.cc` | static registration into core |
| `kernels/` | the op kernels, through the Kernel C API |
