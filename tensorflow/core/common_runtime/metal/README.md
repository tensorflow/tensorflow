# Metal PluggableDevice backend

GPU support for Apple silicon Macs, built into TensorFlow rather than
installed as a separate plugin.

## Status

Experimental. The device and memory foundation, plus enough kernels to train a
convolutional model end to end on the GPU: convolutions and their gradients,
pooling, activations, softmax cross entropy, reductions, weight initialisation
and the Adam and SGD updates.

Coverage is still far from CUDA's. Anything not listed under
[Supported ops](#supported-ops) falls back to the host, which is correct but
slow. See [Limitations](#limitations).

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

Convolutions, pooling, activations, softmax, the cross entropies, batch
normalisation and the reductions go through `MPSGraph`; `MatMul` uses `MPSMatrix` directly because a
2-D multiply needs less machinery.

The `MPSGraph` path is zero-copy in both directions, which is not obvious and
is worth stating: `MPSGraphTensorData`'s `MTLBuffer` initialiser assumes a
tensor starts at the beginning of its buffer, which BFC sub-allocation
guarantees it does not. `MPSNDArray`'s `initWithBuffer:offset:descriptor:`
aliases a buffer at a byte offset, and `MPSGraphTensorData` accepts an
`MPSNDArray`, so operands are fed and results written in place with no staging
buffer anywhere.

Arithmetic goes through `MPSGraph` too, which is what gives it full NumPy
broadcasting rather than the scalar-only broadcasting a hand-written shader
would have to implement by hand.

Random number generation and the optimiser updates are compute shaders rather
than `MPSGraph`. Graphs here are cached by shape, so a seed baked into a graph
would make every call after the first return the same tensor; and MPSGraph
parameterises Adam differently from TensorFlow, which would train subtly
differently rather than fail.

## Supported ops

| Op | dtypes |
| --- | --- |
| `Conv2D`, `Conv2DBackpropInput`, `Conv2DBackpropFilter` | float32, float16 |
| `Conv3D`, `Conv3DBackpropInputV2`, `Conv3DBackpropFilterV2` | float32, float16 |
| `Conv3DBackpropInput`, `Conv3DBackpropFilter` | float32, float16 |
| `MaxPool`, `MaxPoolGrad`, `AvgPool`, `AvgPoolGrad` | float32, float16 |
| `MaxPoolV2`, `MaxPoolGradV2` | float32, float16 |
| `DepthwiseConv2dNative`, `DepthwiseConv2dNativeBackpropInput`, `DepthwiseConv2dNativeBackpropFilter` | float32, float16 |
| `FusedBatchNorm`, `FusedBatchNormV2`, `FusedBatchNormV3` | float32, float16 |
| `FusedBatchNormGrad`, `FusedBatchNormGradV2`, `FusedBatchNormGradV3` | float32, float16 |
| `Relu`, `ReluGrad`, `LeakyRelu`, `LeakyReluGrad` | float32, float16 |
| `Relu6`, `Relu6Grad`, `Softsign`, `SoftsignGrad`, `LogSoftmax` | float32, float16 |
| `Elu`, `EluGrad`, `Selu`, `SeluGrad`, `Softplus`, `SoftplusGrad` | float32, float16 |
| `BatchMatMul`, `BatchMatMulV2`, `BatchMatMulV3` | float32, float16 |
| `BiasAdd`, `BiasAddGrad` | float32, float16 |
| `Softmax` | float32, float16 |
| `SoftmaxCrossEntropyWithLogits` | float32, float16 |
| `SparseSoftmaxCrossEntropyWithLogits` | float32, float16; int32 or int64 labels |
| `MatMul` | float32, float16 |
| `Add`, `AddV2`, `Sub`, `Mul`, `Div`, `RealDiv` | float32, float16 |
| `Maximum`, `Minimum`, `Pow`, `SquaredDifference` | float32, float16 |
| `Neg`, `Abs`, `Square`, `Sqrt`, `Rsqrt`, `Reciprocal` | float32, float16 |
| `Floor`, `Ceil`, `Round`, `Rint`, `Sign`, `Erf` | float32, float16 |
| `Log1p`, `Expm1` | float32, float16 |
| `FloorDiv`, `FloorMod`, `Mod` | float32, float16 |
| `Exp`, `Log`, `Tanh`, `Sigmoid` | float32, float16 |
| `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan` | float32, float16 |
| `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`, `Atan2` | float32, float16 |
| `Xdivy`, `Xlogy` | float32, float16 |
| `TanhGrad`, `SigmoidGrad`, `SqrtGrad`, `RsqrtGrad` | float32, float16 |
| `AddN`, `Transpose`, `Concat`, `ConcatV2`, `Tile` | float32, float16 |
| `Slice`, `Pad`, `PadV2`, `MirrorPad` | float32, float16 |
| `GatherV2`, `OneHot`, `TopKV2` | float32, float16 |
| `Cumsum`, `Cumprod`, `ClipByValue` | float32, float16 |
| `FakeQuantWithMinMaxArgs`, `FakeQuantWithMinMaxArgsGradient` | float32 |
| `MatrixBandPart`, `MatrixDiag`, `MatrixDiagPart`, `MatrixSetDiag` | float32, float16 |
| `MatrixDiagV2`, `MatrixDiagV3`, `MatrixDiagPartV2`, `MatrixDiagPartV3` | float32, float16; main diagonal only |
| `MatrixSetDiagV2`, `MatrixSetDiagV3` | float32, float16; main diagonal only |
| `BatchMatrixBandPart`, `BatchMatrixDiag`, `BatchMatrixDiagPart`, `BatchMatrixSetDiag` | float32, float16 |
| `BiasAddV1`, `ConjugateTranspose`, `Bucketize` | float32, float16 |
| `Conj`, `Cross` | float32, float16 |
| `SpaceToDepth`, `DepthToSpace`, `L2Loss` | float32, float16 |
| `SpaceToBatchND`, `BatchToSpaceND` | float32, float16 |
| `SpaceToBatch`, `BatchToSpace` | float32, float16 |
| `ReverseSequence` | float32, float16 |
| `Diag`, `DiagPart`, `LinSpace` | float32, float16 |
| `ResizeBilinear`, `ResizeNearestNeighbor` | float32, float16 |
| `RGBToHSV`, `HSVToRGB`, `AdjustContrastv2` | float32 |
| `ReverseV2`, `Split`, `SplitV` | float32, float16 |
| `Reverse`, `CheckNumerics`, `CheckNumericsV2` | float32, float16 |
| `ExtractImagePatches` | float32, float16 |
| `LRN` | float32 |
| `StridedSlice`, `StridedSliceGrad`, `TileGrad`, `Roll` | float32, float16 |
| `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual` | float32, float16, int32, int64 |
| `ApproximateEqual` | float32, float16 |
| `LogicalAnd`, `LogicalOr`, `LogicalNot` | bool |
| `Select`, `SelectV2` | float32, float16, int32, int64 |
| `ArgMax`, `ArgMin` | float32, float16; int32 or int64 output |
| `InTopK`, `InTopKV2` | float32 predictions; int32 or int64 targets |
| `Sum`, `Mean`, `Max`, `Min`, `Prod` | float32, float16 |
| `EuclideanNorm` | float32, float16 |
| `Any`, `All` | bool |
| `Fill`, `ZerosLike`, `OnesLike` | float32, float16 |
| `RandomUniform`, `RandomStandardNormal`, `TruncatedNormal` | float32 |
| `RandomUniformInt` | int32 output |
| `ResourceApplyGradientDescent`, `ResourceApplyAdam` | float32 |
| `ResourceApplyMomentum`, `ResourceApplyKerasMomentum`, `ResourceApplyRMSProp` | float32 |
| `Cast` | float32, float16, bfloat16, int32, int64 pairs |
| `Identity` | float32, float16, int32, int64, bool |

Resource variables (`VarHandleOp`, `ReadVariableOp`, `AssignVariableOp` and the
rest), `Reshape`, `Const`, `Shape`, `StridedSlice` and `Pack` need no kernel
here: TensorFlow registers them for `DEVICE_DEFAULT`, which any device type
inherits when it has no kernel of its own.

## Limitations

* **Op coverage is far short of CUDA's**, which has kernels for several
  hundred ops. What is here covers a convolutional classifier and the
  arithmetic around it; anything else falls back to the host, which is correct
  but slow. Notably absent: recurrent layers, `Slice`, `Pad`, `Gather`,
  `DepthwiseConv2d`, and the sparse optimiser variants.
* **Random ops and optimisers are float32 only.**
* **`CheckNumerics` forwards its input without checking.** Detecting a
  non-finite value on device needs a readback of a reduction on every call,
  which would serialise the stream. The values pass through unchanged; the
  op does not raise.
* **`MaxPoolWithArgmax` is not implemented.** MPSGraph returns the position
  within the pooling window rather than the flattened position in the image,
  and emitting indices in the wrong coordinate system would quietly corrupt
  any model that unpools with them.
* **The resize gradients are not implemented.** Every MPSGraph resize
  gradient entry point aborts the process on the current SDK with a channel
  mismatch assertion, for every shape and layout tried. Registering a kernel
  that crashes is worse than leaving the gradient on the host, so the forward
  resizes are provided and the gradients are not.
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
| `kernels/metal_mps_graph.{h,mm}` | MPSGraph bridge, graph cache, zero-copy tensor aliasing |
| `kernels/metal_shader_library.{h,mm}` | Embedded Metal source and pipeline cache |
| `kernels/metal_*_ops.mm` | Op kernels, grouped by family |
