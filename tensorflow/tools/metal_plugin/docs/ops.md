# Metal PluggableDevice backend

GPU support for Apple silicon Macs, built into TensorFlow rather than
installed as a separate plugin.

## Status

Experimental, and complete in the sense that matters for portability: every op
the CUDA build registers for a GPU is registered here too, apart from the
TensorRT ops, which are gated behind `if_tensorrt` and are not part of a macOS
build at all.

Most of those ops run as Metal kernels. The rest are covered the way
TensorFlow already covers them for any device without a kernel of its own,
either by a `DEVICE_DEFAULT` registration or by an unguarded `DEVICE_GPU` one.
A few are registered with their data pinned to host memory, which is correct
but not fast; those are called out under [Limitations](#limitations).

Registering an op is not the same as running it well. What has been measured
end to end is a convolutional classifier: convolutions and their gradients,
pooling, activations, softmax cross entropy, reductions, weight initialisation
and the Adam and SGD updates.

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
| `Conv` | float32, float16 |
| `_FusedConv2D`, `_FusedMatMul` | float32, float16 |
| `_FusedBatchNormEx`, `_FusedBatchNormGradEx` | float32, float16 |
| `CTCLoss`, `CTCLossV2` | float32 |
| `Conv3D`, `Conv3DBackpropInputV2`, `Conv3DBackpropFilterV2` | float32, float16 |
| `Conv3DBackpropInput`, `Conv3DBackpropFilter` | float32, float16 |
| `MaxPool`, `MaxPoolGrad`, `AvgPool`, `AvgPoolGrad` | float32, float16 |
| `MaxPoolV2`, `MaxPoolGradV2` | float32, float16 |
| `MaxPoolWithArgmax`, `MaxPoolGradWithArgmax`, `MaxPoolGradGradWithArgmax` | float32 |
| `MaxPoolGradGrad`, `MaxPoolGradGradV2` | float32 |
| `DepthwiseConv2dNative`, `DepthwiseConv2dNativeBackpropInput`, `DepthwiseConv2dNativeBackpropFilter` | float32, float16 |
| `BatchNormWithGlobalNormalization`, `BatchNormWithGlobalNormalizationGrad` | float32 |
| `Bincount`, `DenseBincount` | float32, int32 |
| `FFT`, `FFT2D`, `FFT3D`, `IFFT`, `IFFT2D`, `IFFT3D` | complex64 |
| `BatchFFT`, `BatchFFT2D`, `BatchFFT3D`, `BatchIFFT`, `BatchIFFT2D`, `BatchIFFT3D` | complex64 |
| `RFFT`, `RFFT2D`, `RFFT3D`, `IRFFT`, `IRFFT2D`, `IRFFT3D` | float32 and complex64 |
| `FFTND`, `IFFTND`, `RFFTND`, `IRFFTND` | complex64, and float32 for the real pair |
| `SparseToDense`, `SparseTensorDenseMatMul` | float32 |
| `SparseBincount`, `RaggedBincount` | float32, int32 |
| `Betainc` | float32 |
| `Snapshot` | float32, float16, int32, int64 |
| `Assign`, `AssignAdd`, `AssignSub` | float32, float16, int32, int64 |
| `DebugNumericSummaryV2` | float32 input and output |
| `_TensorToHashBucketFast` | int8, int16, int32, int64 |
| `NcclAllReduce`, `NcclBroadcast`, `NcclReduce` | float32, float16, float64, int32, int64 |
| `_NcclBroadcastSend`, `_NcclBroadcastRecv`, `_NcclReduceSend`, `_NcclReduceRecv` | float32, float16, float64, int32, int64 |
| `Empty` | float32, int32 |
| `SparseReshape`, `SparseReorder`, `SparseSlice`, `SparseSliceGrad` | float32 |
| `SparseSplit`, `SparseConcat`, `SparseFillEmptyRows`, `SparseFillEmptyRowsGrad` | float32 |
| `RaggedFillEmptyRows`, `RaggedFillEmptyRowsGrad` | float32 |
| `SparseSegmentSum`, `SparseSegmentMean`, `SparseSegmentSqrtN` | float32 |
| `SparseSegmentSumWithNumSegments`, `SparseSegmentMeanWithNumSegments`, `SparseSegmentSqrtNWithNumSegments` | float32 |
| `SparseSegmentSumGrad`, `SparseSegmentMeanGrad`, `SparseSegmentSqrtNGrad` | float32 |
| `SparseSegmentSumGradV2`, `SparseSegmentMeanGradV2`, `SparseSegmentSqrtNGradV2` | float32 |
| `Unique`, `UniqueWithCounts` | float32, int32, int64 |
| `DynamicPartition`, `DynamicStitch`, `ParallelDynamicStitch` | float32, int32, int64 |
| `CropAndResize`, `CropAndResizeGradImage`, `CropAndResizeGradBoxes` | float32 |
| `ImageProjectiveTransformV2`, `ImageProjectiveTransformV3` | float32 |
| `ExtractVolumePatches` | float32 |
| `LSTMBlockCell`, `LSTMBlockCellGrad` | float32 |
| `BlockLSTM`, `BlockLSTMGrad`, `BlockLSTMV2`, `BlockLSTMGradV2` | float32 |
| `GRUBlockCell`, `GRUBlockCellGrad` | float32 |
| `Dilation2D` | float32, float16 |
| `Dilation2DBackpropInput`, `Dilation2DBackpropFilter` | float32 |
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
| `Slice`, `Pad`, `PadV2`, `MirrorPad`, `MirrorPadGrad` | float32, float16 |
| `GatherV2`, `OneHot`, `TopKV2` | float32, float16 |
| `ResourceGather`, `ResourceScatterUpdate` | float32, float16 |
| `GatherNd`, `ResourceGatherNd` | float32, float16 |
| `TopK`, `LowerBound`, `UpperBound`, `HistogramFixedWidth` | float32, float16 |
| `ApproxTopK` | float32, float16 |
| `MatrixTriangularSolve`, `BatchMatrixTriangularSolve` | float32 |
| `Lu` | float32, with int32 or int64 permutation |
| `Qr`, `SelfAdjointEigV2` | float32 |
| `NonMaxSuppressionV2`, `NonMaxSuppressionV3`, `NonMaxSuppressionV4` | float32 |
| `GenerateBoundingBoxProposals` | float32 |
| `_ParallelConcatStart`, `_ParallelConcatUpdate` | float32, float16, int32, int64 |
| `ParallelConcat` | float32, float16, int32, int64; fails if reached, as on every device |
| `Cumsum`, `Cumprod`, `ClipByValue` | float32, float16 |
| `FakeQuantWithMinMaxArgs`, `FakeQuantWithMinMaxArgsGradient` | float32 |
| `FakeQuantWithMinMaxVars`, `FakeQuantWithMinMaxVarsGradient` | float32 |
| `FakeQuantWithMinMaxVarsPerChannel`, `FakeQuantWithMinMaxVarsPerChannelGradient` | float32 |
| `QuantizeAndDequantize`, `QuantizeAndDequantizeV2`, `QuantizeAndDequantizeV3` | float32 |
| `QuantizeAndDequantizeV4`, `QuantizeAndDequantizeV4Grad` | float32 |
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
| `ResizeBilinearGrad`, `ResizeNearestNeighborGrad` | float32 |
| `RGBToHSV`, `HSVToRGB`, `AdjustContrastv2` | float32 |
| `AdjustHue`, `AdjustSaturation` | float32 |
| `ReverseV2`, `Split`, `SplitV` | float32, float16 |
| `Reverse`, `CheckNumerics`, `CheckNumericsV2` | float32, float16 |
| `ExtractImagePatches` | float32, float16 |
| `LRN`, `LRNGrad` | float32 |
| `PopulationCount` | int32, int64 |
| `CumulativeLogsumexp` | float32, float16 |
| `AdjustContrast` | float32 |
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
| `ParameterizedTruncatedNormal`, `StatelessParameterizedTruncatedNormal` | float32 |
| `Multinomial`, `StatelessMultinomial` | float32 logits, int32 or int64 output |
| `RandomGamma`, `StatelessRandomGammaV2`, `StatelessRandomGammaV3` | float32 |
| `RandomUniformInt` | int32 output |
| `ResourceApplyGradientDescent`, `ResourceApplyAdam` | float32 |
| `ResourceApplyMomentum`, `ResourceApplyKerasMomentum`, `ResourceApplyRMSProp` | float32 |
| `Cast` | float32, float16, bfloat16, int32, int64 pairs |
| `Identity` | float32, float16, int32, int64, bool |
| `CudnnRNN`, `CudnnRNNV2`, `CudnnRNNV3` | float32, float16 |
| `CudnnRNNBackprop`, `CudnnRNNBackpropV2`, `CudnnRNNBackpropV3` | float32, float16 |
| `CudnnRNNParamsSize` | float32, float16 |
| `CudnnRNNParamsToCanonical`, `CudnnRNNParamsToCanonicalV2` | float32, float16 |
| `CudnnRNNCanonicalToParams`, `CudnnRNNCanonicalToParamsV2` | float32, float16 |

Resource variables (`VarHandleOp`, `ReadVariableOp`, `AssignVariableOp` and the
rest), `Reshape`, `Const`, `Shape`, `StridedSlice` and `Pack` need no kernel
here: TensorFlow registers them for `DEVICE_DEFAULT`, which any device type
inherits when it has no kernel of its own.

## Limitations

* **Registered is not the same as accelerated.** `TensorArray` and the CSR
  sparse matrix ops are registered with their tensors pinned to host memory,
  because their kernels run the host's arithmetic over a resource the kernel C
  API cannot reach. On a unified memory device the pinning costs a memcpy
  rather than a transfer, but the arithmetic itself is on the CPU. This is the
  same missing C API as
  [#126374](https://github.com/tensorflow/tensorflow/issues/126374) and is not
  fixable from inside a plugin; the proposed fix is
  [#126377](https://github.com/tensorflow/tensorflow/pull/126377).
* **The recurrent parameter buffer's layout is this backend's own.** That is
  allowed because the buffer is opaque and the canonical conversions are the
  only defined way in and out of it, but a checkpoint holding a buffer written
  by cuDNN will not load; one holding canonical weights will. Dropout's masks
  are likewise this backend's own sequence, since nothing outside cuDNN
  defines that one either: what is guaranteed is the rate, the inverted
  scaling, the placement between layers, and reproducibility from the seed.
  Everything else cuDNN accepts is implemented, including `skip_input`, a
  recurrent projection, all four cell types, both directions, any number of
  layers and per-sequence lengths.
* **`ParallelConcat` is registered but always fails**, which is what every
  device does, CUDA included: the graph rewrite replaces the op with an
  allocation and one update per stacked value, so reaching the kernel means
  the rewrite did not run. Both ops it is replaced by are implemented. This is
  correct behaviour rather than a gap.
* **The graph pass fuses a bias and an activation, and nothing else.** A
  folded batch normalisation would need more inputs than the fused kernels
  read, and fusing across a tensor with more than one consumer would leave the
  other consumers pointing at a node that no longer exists, so both are
  refused. The pass also turns TensorFlow's layout optimizer off, because
  MPSGraph takes NHWC and NCHW alike and its transposes are pure loss here.
* **The profiler reports command buffers, not kernels.** One event per
  submission, named after the node that submitted it. A command buffer that
  the runtime issues on its own, a copy or a fill, has no node to name and
  appears as `unnamed`.
* **`MatMul` takes rank-2 tensors only**, which is the op definition rather
  than a restriction: anything higher is `BatchMatMulV2` or `V3`, which this
  backend implements separately.
* **The max pooling ops that carry indices are NHWC only**, which is again the
  op definition: none of them has a `data_format` attribute.
* **Single device.** Every Apple silicon Mac reports one GPU; multi-device has
  had no testing.

## Ops the CUDA build registers and this one does not

The list is the five TensorRT ops, `TRTEngineOp` and the four that manage its
resource. They are gated behind `if_tensorrt`, TensorRT does not build on
macOS, and the ops therefore do not exist in this build to be registered for.

Everything else CUDA registers for a GPU is registered here, by one of four
routes, and it is worth knowing which because they are not equally fast:

* **A Metal kernel**, for most of them. These are the ops in the table above.
* **`DEVICE_DEFAULT`**, TensorFlow's own registration for a device with no
  kernel of its own. `Reshape`, `Const`, `Shape`, the resource variable ops
  and the input pipeline all arrive this way, with no code here.
* **An unguarded `DEVICE_GPU` registration** in TensorFlow's own kernels. The
  Metal platform's device type is `GPU`, so a registration that is not inside
  a `GOOGLE_CUDA` guard already applies to it.
* **`PLUGGABLE_DEVICE_SUPPORTED_MACOS`**, a macro TensorFlow's kernels already
  carry for exactly this situation and that nothing had ever defined. A Metal
  build defines it, which turns on the host-memory registrations for
  `TensorArray` and the CSR sparse matrix ops. That is the pinned-memory case
  under [Limitations](#limitations).

Two groups are registered but do less than the CUDA kernel of the same name,
and say so rather than pretending otherwise:

* **The NCCL collectives** reduce across devices, and every Apple silicon Mac
  reports one GPU. `NcclAllReduce`, `NcclBroadcast` and `NcclReduce` copy their
  input to their output, which is what reducing over one device means, and
  fail at construction when asked for more than one.
* **`ParallelConcat`** fails with TensorFlow's own message, as it does on
  every device including CUDA.

## Files

| File | Role |
| --- | --- |
| `metal_buffer_registry.{h,mm}` | Device address to `MTLBuffer` mapping, allocator stats |
| `metal_stream.{h,mm}` | Streams, events, timers, `OrderedCommandBuffer`, per-device state |
| `metal_stream_executor.{h,mm}` | The `SP_StreamExecutor` callback table |
| `metal_platform.{h,mm}` | `SP_Platform`, device discovery, plugin entry point |
| `metal_profiler.{h,mm}` | The pluggable profiler: op labels, GPU timings, XSpace |
| `metal_graph.{h,mm}` | The graph optimizer: bias and activation fusion |
| `metal_plugin_registrar.cc` | Static registration with core |
| `kernels/metal_mps_graph.{h,mm}` | MPSGraph bridge, graph cache, zero-copy tensor aliasing |
| `kernels/metal_shader_library.{h,mm}` | Embedded Metal source and pipeline cache |
| `kernels/metal_*_ops.mm` | Op kernels, grouped by family |
