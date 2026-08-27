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

#include <algorithm>
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

// StridedSlice and its gradient, TileGrad and Roll.
//
// StridedSlice maps onto MPSGraph unusually cleanly: its
// sliceTensor:starts:ends:strides:startMask:endMask:squeezeMask: takes the
// same three bitmasks TensorFlow calls begin_mask, end_mask and
// shrink_axis_mask, with the same meaning, and sliceGradientTensor is the
// matching reverse. What is not supported is ellipsis_mask and new_axis_mask;
// those rewrite the rank rather than the extents, and are refused rather than
// approximated.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

bool ReadHostVector(TF_Tensor* t, std::vector<int64_t>* out,
                    TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory argument has no data.");
    return false;
  }
  out->clear();
  for (int64_t i = 0; i < count; ++i) {
    if (dtype == TF_INT32) out->push_back(static_cast<const int32_t*>(data)[i]);
    else if (dtype == TF_INT64) out->push_back(static_cast<const int64_t*>(data)[i]);
    else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32 or int64 argument.");
      return false;
    }
  }
  return true;
}

struct StridedOp {
  TF_DataType dtype = TF_FLOAT;
  int32_t begin_mask = 0;
  int32_t end_mask = 0;
  int32_t ellipsis_mask = 0;
  int32_t new_axis_mask = 0;
  int32_t shrink_axis_mask = 0;
};

void* StridedOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new StridedOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  struct { const char* name; int32_t* slot; } masks[] = {
      {"begin_mask", &op->begin_mask},
      {"end_mask", &op->end_mask},
      {"ellipsis_mask", &op->ellipsis_mask},
      {"new_axis_mask", &op->new_axis_mask},
      {"shrink_axis_mask", &op->shrink_axis_mask},
  };
  for (auto& m : masks) {
    TF_OpKernelConstruction_GetAttrInt32(ctx, m.name, m.slot, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      *m.slot = 0;
    }
  }
  TF_DeleteStatus(status);
  return op;
}

void StridedOp_Delete(void* kernel) { delete static_cast<StridedOp*>(kernel); }

// Resolves TensorFlow's begin/end/stride triple into the extents the slice
// actually produces, applying the masks and the negative-index convention.
bool ResolveSlice(const StridedOp& op, const std::vector<int64_t>& in_shape,
                  const std::vector<int64_t>& begin,
                  const std::vector<int64_t>& end,
                  const std::vector<int64_t>& strides,
                  std::vector<int64_t>* out_shape, TF_Status* status) {
  const int rank = static_cast<int>(in_shape.size());
  if (static_cast<int>(begin.size()) != rank ||
      static_cast<int>(end.size()) != rank ||
      static_cast<int>(strides.size()) != rank) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: StridedSlice needs begin, end and strides to cover "
                 "every dimension; a shorter specification implies an "
                 "ellipsis, which is not supported.");
    return false;
  }
  out_shape->clear();
  for (int i = 0; i < rank; ++i) {
    const int64_t stride = strides[i];
    if (stride == 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: StridedSlice stride must not be zero.");
      return false;
    }
    int64_t b = begin[i];
    int64_t e = end[i];
    if (b < 0) b += in_shape[i];
    if (e < 0) e += in_shape[i];
    if (op.begin_mask & (1 << i)) b = stride > 0 ? 0 : in_shape[i] - 1;
    if (op.end_mask & (1 << i)) e = stride > 0 ? in_shape[i] : -1;
    b = std::max<int64_t>(0, std::min<int64_t>(b, in_shape[i]));
    if (stride > 0) e = std::max<int64_t>(0, std::min<int64_t>(e, in_shape[i]));

    if (op.shrink_axis_mask & (1 << i)) continue;  // squeezed away
    int64_t extent = 0;
    if (stride > 0) {
      extent = e > b ? (e - b + stride - 1) / stride : 0;
    } else {
      extent = b > e ? (b - e + (-stride) - 1) / (-stride) : 0;
    }
    out_shape->push_back(extent);
  }
  return true;
}

NSArray<NSNumber*>* ToNS(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

/*** STRIDED SLICE ***/

void StridedSlice_ComputeImpl(StridedOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  if (op->ellipsis_mask != 0 || op->new_axis_mask != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: StridedSlice with ellipsis_mask or new_axis_mask is "
                 "not supported; those change the rank rather than the "
                 "extents.");
    return;
  }

  ScopedTensor input, begin_t, end_t, strides_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, begin_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, end_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, strides_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  std::vector<int64_t> begin, end, strides, out_shape;
  if (!ReadHostVector(begin_t.get(), &begin, status)) return;
  if (!ReadHostVector(end_t.get(), &end, status)) return;
  if (!ReadHostVector(strides_t.get(), &strides, status)) return;
  if (!ResolveSlice(*op, in_shape, begin, end, strides, &out_shape, status)) {
    return;
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "StridedSlice";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(begin, &key);
  AppendShapeToKey(end, &key);
  AppendShapeToKey(strides, &key);
  key.append("/m").append(std::to_string(op->begin_mask)).push_back(',');
  key.append(std::to_string(op->end_mask)).push_back(',');
  key.append(std::to_string(op->shrink_axis_mask));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSArray<NSNumber*>* starts = ToNS(begin);
  NSArray<NSNumber*>* ends = ToNS(end);
  NSArray<NSNumber*>* steps = ToNS(strides);
  const uint32_t bm = static_cast<uint32_t>(op->begin_mask);
  const uint32_t em = static_cast<uint32_t>(op->end_mask);
  const uint32_t sm = static_cast<uint32_t>(op->shrink_axis_mask);
  const std::vector<int64_t> final_shape = out_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* r = [out->graph sliceTensor:x
                                             starts:starts
                                               ends:ends
                                            strides:steps
                                          startMask:bm
                                            endMask:em
                                        squeezeMask:sm
                                               name:nil];
        [out->inputs addObject:x];
        // MPSGraph's squeeze and TensorFlow's shrink agree, but reshaping to
        // the shape core allocated keeps the two definitions from drifting.
        [out->outputs addObject:[out->graph reshapeTensor:r
                                                withShape:MPSShape(final_shape)
                                                     name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** STRIDED SLICE GRADIENT ***/

void StridedSliceGrad_ComputeImpl(StridedOp* op, TF_OpKernelContext* ctx,
                                  TF_Status* status) {
  if (op->ellipsis_mask != 0 || op->new_axis_mask != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: StridedSliceGrad with ellipsis_mask or new_axis_mask "
                 "is not supported.");
    return;
  }

  ScopedTensor shape_t, begin_t, end_t, strides_t, grad;
  TF_GetInput(ctx, 0, shape_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, begin_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, end_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, strides_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> out_shape, begin, end, strides, sliced;
  if (!ReadHostVector(shape_t.get(), &out_shape, status)) return;
  if (!ReadHostVector(begin_t.get(), &begin, status)) return;
  if (!ReadHostVector(end_t.get(), &end, status)) return;
  if (!ReadHostVector(strides_t.get(), &strides, status)) return;
  if (!ResolveSlice(*op, out_shape, begin, end, strides, &sliced, status)) {
    return;
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  std::string key = "StridedSliceGrad";
  AppendShapeToKey(out_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendShapeToKey(begin, &key);
  AppendShapeToKey(end, &key);
  AppendShapeToKey(strides, &key);
  key.append("/m").append(std::to_string(op->begin_mask)).push_back(',');
  key.append(std::to_string(op->end_mask)).push_back(',');
  key.append(std::to_string(op->shrink_axis_mask));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSArray<NSNumber*>* starts = ToNS(begin);
  NSArray<NSNumber*>* ends = ToNS(end);
  NSArray<NSNumber*>* steps = ToNS(strides);
  const uint32_t bm = static_cast<uint32_t>(op->begin_mask);
  const uint32_t em = static_cast<uint32_t>(op->end_mask);
  const uint32_t sm = static_cast<uint32_t>(op->shrink_axis_mask);
  // sliceGradient needs the forward input shape as a tensor, not an array.
  std::vector<int32_t> shape_values;
  for (int64_t d : out_shape) shape_values.push_back(static_cast<int32_t>(d));
  NSData* shape_data = [NSData dataWithBytes:shape_values.data()
                                      length:shape_values.size() * sizeof(int32_t)];
  const std::vector<int64_t> shape_tensor_shape = {
      static_cast<int64_t>(out_shape.size())};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(grad_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* fwd_shape =
            [g constantWithData:shape_data
                          shape:MPSShape(shape_tensor_shape)
                       dataType:MPSDataTypeInt32];
        [out->inputs addObject:dy];
        [out->outputs addObject:[g sliceGradientTensor:dy
                                      fwdInShapeTensor:fwd_shape
                                                starts:starts
                                                  ends:ends
                                               strides:steps
                                             startMask:bm
                                               endMask:em
                                           squeezeMask:sm
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), op->dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data ], @[ o_data ], status);
}

/*** TILE GRADIENT ***/

// The gradient of Tile sums each repeated block back onto the original
// extent. Reshaping the tiled axis into [multiple, extent] and reducing the
// first of the pair does that without any scatter.
void TileGrad_ComputeImpl(StridedOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor grad, multiples_t;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, multiples_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  const int rank = static_cast<int>(grad_shape.size());
  std::vector<int64_t> mult;
  if (!ReadHostVector(multiples_t.get(), &mult, status)) return;
  if (static_cast<int>(mult.size()) != rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: TileGrad multiples must match the rank.");
    return;
  }

  std::vector<int64_t> out_shape(rank);
  std::vector<int64_t> split_shape;   // [m0, n0, m1, n1, ...]
  NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
  for (int i = 0; i < rank; ++i) {
    if (mult[i] <= 0 || grad_shape[i] % mult[i] != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: TileGrad multiples must divide the gradient.");
      return;
    }
    out_shape[i] = grad_shape[i] / mult[i];
    [reduce_axes addObject:@(static_cast<NSInteger>(split_shape.size()))];
    split_shape.push_back(mult[i]);
    split_shape.push_back(out_shape[i]);
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

  std::string key = "TileGrad";
  AppendShapeToKey(grad_shape, &key);
  AppendShapeToKey(mult, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const std::vector<int64_t> final_shape = out_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(grad_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* split = [g reshapeTensor:dy
                                       withShape:MPSShape(split_shape)
                                            name:nil];
        MPSGraphTensor* summed = [g reductionSumWithTensor:split
                                                      axes:reduce_axes
                                                      name:nil];
        [out->inputs addObject:dy];
        [out->outputs addObject:[g reshapeTensor:summed
                                       withShape:MPSShape(final_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), op->dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data ], @[ o_data ], status);
}

/*** ROLL ***/

// Roll is a concatenation of two slices per axis: the tail moves to the front
// and the head follows it.
void Roll_ComputeImpl(StridedOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input, shift_t, axis_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, shift_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, axis_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int rank = static_cast<int>(shape.size());
  std::vector<int64_t> shifts, axes;
  if (!ReadHostVector(shift_t.get(), &shifts, status)) return;
  if (!ReadHostVector(axis_t.get(), &axes, status)) return;
  if (shifts.size() != axes.size()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Roll needs one shift per axis.");
    return;
  }
  // Fold repeated axes together so each is rotated once.
  std::vector<int64_t> total(rank, 0);
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t a = axes[i];
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Roll axis is out of range.");
      return;
    }
    total[a] += shifts[i];
  }

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Roll";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(total, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* r = x;
        for (int i = 0; i < rank; ++i) {
          const int64_t n = shape[i];
          if (n == 0) continue;
          int64_t s = total[i] % n;
          if (s < 0) s += n;
          if (s == 0) continue;
          // The last s entries wrap round to the front.
          MPSGraphTensor* tail = [g sliceTensor:r
                                      dimension:static_cast<NSUInteger>(i)
                                          start:static_cast<NSInteger>(n - s)
                                         length:static_cast<NSInteger>(s)
                                           name:nil];
          MPSGraphTensor* head = [g sliceTensor:r
                                      dimension:static_cast<NSUInteger>(i)
                                          start:0
                                         length:static_cast<NSInteger>(n - s)
                                           name:nil];
          r = [g concatTensors:@[ tail, head ]
                     dimension:static_cast<NSInteger>(i)
                          name:nil];
        }
        [out->inputs addObject:x];
        [out->outputs addObject:r];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<StridedOp*>(kernel);                               \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(StridedSlice_Compute, StridedSlice_ComputeImpl)
METAL_COMPUTE(StridedSliceGrad_Compute, StridedSliceGrad_ComputeImpl)
METAL_COMPUTE(TileGrad_Compute, TileGrad_ComputeImpl)
METAL_COMPUTE(Roll_Compute, Roll_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_INT32) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &StridedOp_Create, compute, &StridedOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && attr2 != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, attr2, dtype2, status);
  }
  for (const char* a : host_args) TF_KernelBuilder_HostMemory(builder, a);
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

void RegisterMetalStridedKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const std::string s = kSuffixes[i];
    for (int j = 0; j < 2; ++j) {
      const std::string is = kIndexSuffixes[j];
      // Every index argument is read on the host to work out the extents.
      Register("StridedSlice", &StridedSlice_Compute, kDTypes[i],
               "MetalStridedSlice" + s + is, {"begin", "end", "strides"},
               "Index", kIndexTypes[j]);
      Register("StridedSliceGrad", &StridedSliceGrad_Compute, kDTypes[i],
               "MetalStridedSliceGrad" + s + is,
               {"shape", "begin", "end", "strides"}, "Index", kIndexTypes[j]);
      Register("TileGrad", &TileGrad_Compute, kDTypes[i],
               "MetalTileGrad" + s + is, {"multiples"}, "Tmultiples",
               kIndexTypes[j]);
      Register("Roll", &Roll_Compute, kDTypes[i], "MetalRoll" + s + is,
               {"shift", "axis"}, "Tshift", kIndexTypes[j]);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
