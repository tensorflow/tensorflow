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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Transpose, AddN and average pooling.
//
// AddN and Transpose are not optional extras. AddN is how TensorFlow sums the
// gradients reaching a tensor used more than once, which happens in any model
// with a shared or reused layer, and Transpose appears throughout gradient
// graphs. TensorFlow's DEVICE_DEFAULT registrations for both cover only int32
// in host memory, so without these a float model drops to the host at exactly
// the points where its tensors are largest.

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

struct DTypeOp {
  TF_DataType dtype = TF_FLOAT;
};

void* DTypeOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DTypeOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void DTypeOp_Delete(void* kernel) { delete static_cast<DTypeOp*>(kernel); }

/*** TRANSPOSE ***/

void Transpose_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input;
  ScopedTensor perm_tensor;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, perm_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());

  // The permutation is in host memory, so it can be read to build the output
  // shape without draining the stream.
  const int64_t perm_count = TF_TensorElementCount(perm_tensor.get());
  if (perm_count != rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Transpose permutation length does not match the rank.");
    return;
  }
  const TF_DataType perm_dtype = TF_TensorType(perm_tensor.get());
  const void* perm_data = TF_TensorData(perm_tensor.get());
  std::vector<int> perm(rank);
  std::vector<bool> seen(rank, false);
  for (int i = 0; i < rank; ++i) {
    int64_t axis = perm_dtype == TF_INT32
                       ? static_cast<const int32_t*>(perm_data)[i]
                       : static_cast<const int64_t*>(perm_data)[i];
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank || seen[axis]) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Transpose permutation is not a permutation.");
      return;
    }
    seen[axis] = true;
    perm[i] = static_cast<int>(axis);
  }

  std::vector<int64_t> out_shape(rank);
  for (int i = 0; i < rank; ++i) out_shape[i] = in_shape[perm[i]];

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Transpose";
  AppendShapeToKey(in_shape, &key);
  key.push_back('/');
  for (int axis : perm) key.append(std::to_string(axis)).push_back(',');
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSMutableArray<NSNumber*>* permutation = [NSMutableArray array];
  for (int axis : perm) [permutation addObject:@(static_cast<NSInteger>(axis))];

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph transposeTensor:x
                                                permutation:permutation
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** ADD N ***/

void AddN_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  const int n = TF_NumInputs(ctx);
  if (n < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: AddN needs an input.");
    return;
  }

  std::vector<ScopedTensor> inputs(n);
  for (int i = 0; i < n; ++i) {
    TF_GetInput(ctx, i, inputs[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> shape = ShapeOf(inputs[0].get());
  for (int i = 1; i < n; ++i) {
    if (ShapeOf(inputs[i].get()) != shape) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: AddN inputs must all have the same shape.");
      return;
    }
  }

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "AddN/" + std::to_string(n);
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* total = nil;
        for (int i = 0; i < n; ++i) {
          MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                      dataType:mps_dtype
                                                          name:nil];
          [out->inputs addObject:x];
          total = total == nil
                      ? x
                      : [out->graph additionWithPrimaryTensor:total
                                              secondaryTensor:x
                                                         name:nil];
        }
        // A single input still has to produce a distinct result tensor, since
        // the output buffer is separate from the input's.
        if (n == 1) total = [out->graph identityWithTensor:total name:nil];
        [out->outputs addObject:total];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int i = 0; i < n; ++i) {
    MPSGraphTensorData* d =
        TensorDataForTensor(inputs[i].get(), op->dtype, device, status);
    if (d == nil) return;
    [feeds addObject:d];
  }
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ out_data ], status);
}

/*** CONCAT ***/

// ConcatV2 puts the axis last and Concat puts it first; both hold it in host
// memory. `kAxisFirst` selects which convention this instantiation reads.
template <bool kAxisFirst>
void Concat_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  const int total = TF_NumInputs(ctx);
  if (total < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Concat needs at least one value and an axis.");
    return;
  }
  const int first_value = kAxisFirst ? 1 : 0;
  const int axis_index = kAxisFirst ? 0 : total - 1;
  const int n = total - 1;

  ScopedTensor axis_tensor;
  TF_GetInput(ctx, axis_index, axis_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<ScopedTensor> values(n);
  for (int i = 0; i < n; ++i) {
    TF_GetInput(ctx, first_value + i, values[i].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> first_shape = ShapeOf(values[0].get());
  const int rank = static_cast<int>(first_shape.size());
  const void* axis_data = TF_TensorData(axis_tensor.get());
  int64_t axis = TF_TensorType(axis_tensor.get()) == TF_INT32
                     ? *static_cast<const int32_t*>(axis_data)
                     : *static_cast<const int64_t*>(axis_data);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Concat axis is out of range.");
    return;
  }

  // Every value must agree on every axis but the concatenated one.
  std::vector<int64_t> out_shape = first_shape;
  out_shape[axis] = 0;
  std::vector<std::vector<int64_t>> shapes(n);
  for (int i = 0; i < n; ++i) {
    shapes[i] = ShapeOf(values[i].get());
    if (static_cast<int>(shapes[i].size()) != rank) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Concat inputs differ in rank.");
      return;
    }
    for (int d = 0; d < rank; ++d) {
      if (d != axis && shapes[i][d] != first_shape[d]) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     "Metal: Concat inputs differ outside the concat axis.");
        return;
      }
    }
    out_shape[axis] += shapes[i][axis];
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Concat/" + std::to_string(n) + "/a" + std::to_string(axis);
  for (const auto& sh : shapes) AppendShapeToKey(sh, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSInteger mps_axis = static_cast<NSInteger>(axis);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray array];
        for (int i = 0; i < n; ++i) {
          MPSGraphTensor* t = [out->graph placeholderWithShape:MPSShape(shapes[i])
                                                      dataType:mps_dtype
                                                          name:nil];
          [out->inputs addObject:t];
          [parts addObject:t];
        }
        [out->outputs addObject:[out->graph concatTensors:parts
                                                dimension:mps_axis
                                                     name:nil]];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  for (int i = 0; i < n; ++i) {
    MPSGraphTensorData* d =
        TensorDataForTensor(values[i].get(), op->dtype, device, status);
    if (d == nil) return;
    [feeds addObject:d];
  }
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ out_data ], status);
}

/*** TILE ***/

void Tile_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input, multiples;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, multiples.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  if (TF_TensorElementCount(multiples.get()) != rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Tile multiples length does not match the rank.");
    return;
  }
  const void* mult_data = TF_TensorData(multiples.get());
  const bool is32 = TF_TensorType(multiples.get()) == TF_INT32;
  std::vector<int64_t> mult(rank), out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    mult[i] = is32 ? static_cast<const int32_t*>(mult_data)[i]
                   : static_cast<const int64_t*>(mult_data)[i];
    if (mult[i] < 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Tile multiples must not be negative.");
      return;
    }
    out_shape[i] = in_shape[i] * mult[i];
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Tile";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(mult, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  NSMutableArray<NSNumber*>* multiplier = [NSMutableArray array];
  for (int64_t m : mult) [multiplier addObject:@(static_cast<NSInteger>(m))];

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph tileTensor:x
                                        withMultiplier:multiplier
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** AVERAGE POOLING ***/

struct AvgPoolOp {
  SpatialParams params;
  int window_h = 1;
  int window_w = 1;
};

void* AvgPoolOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new AvgPoolOp();
  if (!ReadSpatialParams(ctx, /*want_dilations=*/false, &op->params, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  int32_t ksize[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksize", ksize, 4, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->window_h = ksize[SpatialHeightIndex(op->params.nhwc)];
  op->window_w = ksize[SpatialWidthIndex(op->params.nhwc)];
  TF_DeleteStatus(status);
  return op;
}

void AvgPoolOp_Delete(void* kernel) { delete static_cast<AvgPoolOp*>(kernel); }

MPSGraphPooling2DOpDescriptor* AvgDescriptorFor(const AvgPoolOp& op) {
  return [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:static_cast<NSUInteger>(op.window_w)
                   kernelHeight:static_cast<NSUInteger>(op.window_h)
                      strideInX:static_cast<NSUInteger>(op.params.stride_w)
                      strideInY:static_cast<NSUInteger>(op.params.stride_h)
                   paddingStyle:op.params.same_padding
                                    ? MPSGraphPaddingStyleTF_SAME
                                    : MPSGraphPaddingStyleTF_VALID
                     dataLayout:op.params.nhwc
                                    ? MPSGraphTensorNamedDataLayoutNHWC
                                    : MPSGraphTensorNamedDataLayoutNCHW];
}

int64_t PoolExtent(int64_t input, int window, int stride, bool same) {
  if (same) return (input + stride - 1) / stride;
  if (input < window) return 0;
  return (input - window) / stride + 1;
}

void AvgPool_ComputeImpl(AvgPoolOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: AvgPool expects a rank-4 input.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int h = SpatialHeightIndex(op->params.nhwc);
  const int w = SpatialWidthIndex(op->params.nhwc);
  std::vector<int64_t> out_shape = in_shape;
  out_shape[h] = PoolExtent(in_shape[h], op->window_h, op->params.stride_h,
                            op->params.same_padding);
  out_shape[w] = PoolExtent(in_shape[w], op->window_w, op->params.stride_w,
                            op->params.same_padding);

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->params.dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->params.dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;

  std::string key = "AvgPool";
  AppendShapeToKey(in_shape, &key);
  key.append("/k").append(std::to_string(op->window_h)).push_back('x');
  key.append(std::to_string(op->window_w));
  key.append("/s").append(std::to_string(op->params.stride_h)).push_back('x');
  key.append(std::to_string(op->params.stride_w));
  key.append(op->params.same_padding ? "/SAME" : "/VALID");
  key.append(op->params.nhwc ? "/NHWC" : "/NCHW");
  key.append("/t").append(std::to_string(static_cast<int>(op->params.dtype)));
  const AvgPoolOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* src = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                      dataType:mps_dtype
                                                          name:nil];
        [out->inputs addObject:src];
        [out->outputs addObject:[out->graph
                                    avgPooling2DWithSourceTensor:src
                                                      descriptor:AvgDescriptorFor(
                                                                     captured)
                                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->params.dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, STATE, IMPL)                                      \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<STATE*>(kernel);                                   \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(Transpose_Compute, DTypeOp, Transpose_ComputeImpl)
METAL_COMPUTE(AddN_Compute, DTypeOp, AddN_ComputeImpl)
METAL_COMPUTE(AvgPool_Compute, AvgPoolOp, AvgPool_ComputeImpl)
METAL_COMPUTE(ConcatV2_Compute, DTypeOp, Concat_ComputeImpl<false>)
METAL_COMPUTE(Concat_Compute, DTypeOp, Concat_ComputeImpl<true>)
METAL_COMPUTE(Tile_Compute, DTypeOp, Tile_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*),
              void (*destroy)(void*), TF_DataType dtype,
              const std::string& name, const char* host_arg = nullptr,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_INT32) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && attr2 != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, attr2, dtype2, status);
  }
  if (host_arg != nullptr) TF_KernelBuilder_HostMemory(builder, host_arg);
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalArrayKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kPermTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kPermSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const std::string suffix = kSuffixes[i];
    for (int j = 0; j < 2; ++j) {
      // The permutation is read on the host to build the output shape.
      Register("Transpose", &DTypeOp_Create, &Transpose_Compute,
               &DTypeOp_Delete, kDTypes[i],
               "MetalTranspose" + suffix + kPermSuffixes[j], "perm", "Tperm",
               kPermTypes[j]);
    }
    Register("AddN", &DTypeOp_Create, &AddN_Compute, &DTypeOp_Delete,
             kDTypes[i], "MetalAddN" + suffix);
    Register("AvgPool", &AvgPoolOp_Create, &AvgPool_Compute, &AvgPoolOp_Delete,
             kDTypes[i], "MetalAvgPool" + suffix);
    // The concat axis and the tile multiples are read on the host to size the
    // output, so both stay off the device.
    Register("ConcatV2", &DTypeOp_Create, &ConcatV2_Compute, &DTypeOp_Delete,
             kDTypes[i], "MetalConcatV2" + suffix, "axis");
    Register("Concat", &DTypeOp_Create, &Concat_Compute, &DTypeOp_Delete,
             kDTypes[i], "MetalConcat" + suffix, "concat_dim");
    Register("Tile", &DTypeOp_Create, &Tile_Compute, &DTypeOp_Delete,
             kDTypes[i], "MetalTile" + suffix, "multiples");
  }
}

}  // namespace metal
}  // namespace tensorflow
