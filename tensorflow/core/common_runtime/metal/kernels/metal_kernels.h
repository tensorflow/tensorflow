/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_

// Names no Objective-C types; includable from plain C++.

namespace tensorflow {
namespace metal {

// Elementwise arithmetic (Add, AddV2, Sub, Mul) and Cast, for float32 and
// float16, implemented as Metal compute shaders.
void RegisterMetalElementwiseKernels();

// MatMul for float32 and float16, backed by MPSMatrixMultiplication.
void RegisterMetalMatMulKernels();

// Conv2D and its gradients with respect to the input and the filter, on
// MPSGraph, for float32 and float16.
void RegisterMetalConvKernels();

// Relu, BiasAdd, Softmax, the dense and sparse softmax cross entropies, and
// the gradients of each, on MPSGraph.
void RegisterMetalNnKernels();

// MaxPool and its gradient, on MPSGraph.
void RegisterMetalPoolingKernels();

// Sum and Mean over float tensors, on MPSGraph. TensorFlow's DEVICE_DEFAULT
// registrations for these cover int32 in host memory only.
void RegisterMetalReductionKernels();

// Fill, ZerosLike and OnesLike over float tensors, which TensorFlow also
// registers generically for int32 in host memory only.
void RegisterMetalFillKernels();

// RandomUniform, RandomStandardNormal and TruncatedNormal, which is what
// initialises a model's weights before the first step.
void RegisterMetalRandomKernels();

// ResourceApplyGradientDescent and ResourceApplyAdam, the weight update that
// closes a training step.
void RegisterMetalTrainingKernels();

// Transpose, AddN and AvgPool over float tensors. TensorFlow registers the
// first two generically for int32 in host memory only.
void RegisterMetalArrayKernels();

// FusedBatchNorm v1, v2 and v3 with their gradients, in both training and
// inference mode.
void RegisterMetalBatchNormKernels();

// Comparisons, the logical operators, Select and ArgMax/ArgMin, which is what
// an accuracy metric is built from.
void RegisterMetalCompareKernels();

// LeakyRelu, the activation gradients that carry an alpha or a scale, and
// BatchMatMul v1 through v3.
void RegisterMetalActivationKernels();

// Slice, Pad, MirrorPad, ReverseV2, Split and SplitV over float tensors.
void RegisterMetalSliceKernels();

// GatherV2, OneHot, TopKV2, the cumulative scans and ClipByValue.
void RegisterMetalIndexKernels();

// MatrixBandPart, the diagonal family, L2Loss and the space/depth
// rearrangements.
void RegisterMetalMatrixKernels();

// StridedSlice and its gradient, TileGrad and Roll.
void RegisterMetalStridedKernels();

// Depthwise convolution with both gradients, and the average pooling
// gradient.
void RegisterMetalDepthwiseKernels();

// Image resizing, bilinear and nearest, with both gradients.
void RegisterMetalImageKernels();

// MaxPoolV2, MaxPoolGradV2 and MaxPoolWithArgmax.
void RegisterMetalPoolVariantKernels();

// Conv3D and its two gradients.
void RegisterMetalConv3DKernels();

// SpaceToBatchND and BatchToSpaceND.
void RegisterMetalBatchSpaceKernels();

// FakeQuantWithMinMaxArgs and its gradient.
void RegisterMetalQuantKernels();

// Reverse, LRN and CheckNumerics.
void RegisterMetalMiscKernels();

// BiasAddV1, ConjugateTranspose and Bucketize.
void RegisterMetalAliasKernels();

// RGBToHSV and AdjustContrastv2.
void RegisterMetalImage2Kernels();

// LowerBound, UpperBound, HistogramFixedWidth and TopK.
void RegisterMetalSearchKernels();

// PopulationCount, CumulativeLogsumexp, LRNGrad and AdjustContrast.
void RegisterMetalExtraKernels();

// ResourceGather and ResourceScatterUpdate.
void RegisterMetalResourceKernels();

// Max pooling with indices, and the second-order pooling gradients.
void RegisterMetalMaxPoolArgmaxKernels();

// QuantizeAndDequantize and its V2, V3 and V4 forms.
void RegisterMetalQuantizeDequantizeKernels();

// ImageProjectiveTransformV2 and V3.
void RegisterMetalTransformKernels();

// CropAndResize and its image and box gradients.
void RegisterMetalCropResizeKernels();

// The parameterised random distributions and their stateless forms.
void RegisterMetalRandomDistKernels();

// The single-step recurrent cells and their gradients.
void RegisterMetalRnnKernels();

// The ops whose output shape depends on their input values.
void RegisterMetalDynamicKernels();

// The two ops a parallel stack decomposes into.
void RegisterMetalInplaceKernels();

// MatrixTriangularSolve and its deprecated alias.
void RegisterMetalLinalgKernels();

// Non-maximum suppression, V2 through V4.
void RegisterMetalNmsKernels();

// Conv, the rank-agnostic convolution.
void RegisterMetalGenericConvKernels();

// CTCLoss and CTCLossV2.
void RegisterMetalCtcKernels();

// GenerateBoundingBoxProposals.
void RegisterMetalBoxProposalKernels();

// The sparse segment reductions and their gradients.
void RegisterMetalSparseSegmentKernels();

// The Fourier transforms.
void RegisterMetalFftKernels();

// The fused convolution and matrix multiply the optimiser produces.
void RegisterMetalFusedKernels();

// The sparse tensor manipulations.
void RegisterMetalSparseManipKernels();

// Betainc, the sparse and ragged bin counts, Snapshot and Empty.
void RegisterMetalMisc2Kernels();

// Assign, AssignAdd and AssignSub on reference variables.
void RegisterMetalRefVariableKernels();

// The NCCL collectives, over one device.
void RegisterMetalCollectiveKernels();

// The CudnnRNN family: the recurrent networks, their parameter buffer and
// its two canonical conversions.
void RegisterMetalCudnnRnnKernels();

// GatherNd, which is not a resource op and is registered on its own.
void RegisterMetalGatherNdKernels();

// DebugNumericSummaryV2 and _TensorToHashBucketFast.
void RegisterMetalDebugKernels();

// SparseToDense and SparseTensorDenseMatMul.
void RegisterMetalSparseKernels();

// ExtractVolumePatches.
void RegisterMetalVolumePatchKernels();

// ResizeBilinearGrad and ResizeNearestNeighborGrad.
void RegisterMetalResizeGradKernels();

// BatchNormWithGlobalNormalization and its gradient.
void RegisterMetalBatchNormGlobalKernels();

// Bincount and DenseBincount.
void RegisterMetalBincountKernels();

// Dilation2D, grayscale morphological dilation.
void RegisterMetalDilationKernels();

// Identity, which aliases its input when it can and blits when it cannot.
void RegisterMetalIdentityKernels();

// Registers every Metal kernel. Passed to core as the plugin's kernel module
// (PluggableDeviceInit_Api::init_kernel_fn), so core decides when kernel
// registration happens relative to device registration rather than the order
// falling out of static initialiser order.
void RegisterAllMetalKernels();

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNELS_H_
