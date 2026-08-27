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
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
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

// Slice, Pad, Split, Reverse, ClipByValue and the space/depth rearrangements.
//
// TensorFlow does register Slice and Pad for DEVICE_DEFAULT, but only for
// int32 with every argument in host memory, exactly as it does for Sum. A
// float slice or pad on device therefore has to be provided here, and these
// are not rare ops: slicing appears in every gradient of a concatenation and
// padding in every convolution written with explicit borders.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Reads a host-memory int32 or int64 vector, which is how TensorFlow passes
// begins, sizes, paddings, axes and split points.
bool ReadHostVector(TF_Tensor* t, std::vector<int64_t>* out,
                    TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory shape argument has no data.");
    return false;
  }
  out->clear();
  out->reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    if (dtype == TF_INT32) {
      out->push_back(static_cast<const int32_t*>(data)[i]);
    } else if (dtype == TF_INT64) {
      out->push_back(static_cast<const int64_t*>(data)[i]);
    } else {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: expected an int32 or int64 shape argument.");
      return false;
    }
  }
  return true;
}

NSArray<NSNumber*>* ToNSArray(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

struct DTypeOp {
  TF_DataType dtype = TF_FLOAT;
  int num_split = 1;
  // MirrorPad's mode; Pad and PadV2 leave it constant.
  MPSGraphPaddingMode pad_mode = MPSGraphPaddingModeConstant;
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
  int32_t num_split = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_split", &num_split, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->num_split = num_split;

  char mode[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "mode", mode, sizeof(mode) - 1,
                                        status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
  } else if (std::strcmp(mode, "REFLECT") == 0) {
    op->pad_mode = MPSGraphPaddingModeReflect;
  } else if (std::strcmp(mode, "SYMMETRIC") == 0) {
    op->pad_mode = MPSGraphPaddingModeSymmetric;
  }
  TF_DeleteStatus(status);
  return op;
}

void DTypeOp_Delete(void* kernel) { delete static_cast<DTypeOp*>(kernel); }

/*** SLICE ***/

void Slice_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor input, begin_t, size_t_;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, begin_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, size_t_.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  std::vector<int64_t> begin, size;
  if (!ReadHostVector(begin_t.get(), &begin, status)) return;
  if (!ReadHostVector(size_t_.get(), &size, status)) return;
  if (static_cast<int>(begin.size()) != rank ||
      static_cast<int>(size.size()) != rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Slice begin and size must match the input rank.");
    return;
  }

  for (int i = 0; i < rank; ++i) {
    // A size of -1 means "everything left on this axis", which is how
    // TensorFlow writes an open-ended slice.
    if (size[i] < 0) size[i] = in_shape[i] - begin[i];
    if (begin[i] < 0 || size[i] < 0 || begin[i] + size[i] > in_shape[i]) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Slice range falls outside the input.");
      return;
    }
  }

  const int64_t count = ElementCount(size);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, size.data(), rank,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Slice";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(begin, &key);
  AppendShapeToKey(size, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* r = x;
        // One dimension at a time; MPSGraph folds the chain into a single
        // strided read rather than materialising each step.
        for (int i = 0; i < rank; ++i) {
          if (begin[i] == 0 && size[i] == in_shape[i]) continue;
          r = [out->graph sliceTensor:r
                            dimension:static_cast<NSUInteger>(i)
                                start:static_cast<NSInteger>(begin[i])
                               length:static_cast<NSInteger>(size[i])
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
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** PAD ***/

template <bool kHasConstant>
void Pad_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input, paddings, constant;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, paddings.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (kHasConstant) {
    TF_GetInput(ctx, 2, constant.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  std::vector<int64_t> flat;
  if (!ReadHostVector(paddings.get(), &flat, status)) return;
  if (static_cast<int>(flat.size()) != rank * 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Pad expects a [rank, 2] paddings tensor.");
    return;
  }

  std::vector<int64_t> left(rank), right(rank), out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    left[i] = flat[i * 2];
    right[i] = flat[i * 2 + 1];
    if (left[i] < 0 || right[i] < 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Pad amounts must not be negative.");
      return;
    }
    out_shape[i] = in_shape[i] + left[i] + right[i];
  }

  // PadV2's fill value lives on the device, but MPSGraph's padTensor takes it
  // as a plain double rather than a tensor. Reading the one scalar back costs
  // a drain of this stream; that is acceptable because a Pad constant is
  // almost always a compile-time zero, and the alternative, building a
  // fill-and-overwrite graph, would cost more on every call.
  double constant_value = 0.0;
  if (kHasConstant) {
    if (op->dtype != TF_FLOAT) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: PadV2 reads its fill value on the host and supports "
                   "float32 only; pad with zero and cast instead.");
      return;
    }
    SP_Stream drain = StreamForContext(ctx, status);
    if (TF_GetCode(status) != TF_OK) return;
    uint64_t target = 0;
    {
      absl::MutexLock lock(&drain->mu);
      target = drain->last_enqueued;
    }
    if (target > 0) {
      [drain->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
    }
    const void* p = TF_TensorData(constant.get());
    if (p == nullptr) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: PadV2 constant has no data.");
      return;
    }
    constant_value = static_cast<double>(*static_cast<const float*>(p));
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

  std::string key = "Pad";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(left, &key);
  AppendShapeToKey(right, &key);
  key.append("/m").append(std::to_string(static_cast<int>(op->pad_mode)));
  key.append("/c").append(std::to_string(constant_value));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSMutableArray<NSNumber*>* lpad = [NSMutableArray array];
  NSMutableArray<NSNumber*>* rpad = [NSMutableArray array];
  for (int i = 0; i < rank; ++i) {
    [lpad addObject:@(static_cast<NSInteger>(left[i]))];
    [rpad addObject:@(static_cast<NSInteger>(right[i]))];
  }
  const MPSGraphPaddingMode mode = op->pad_mode;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph padTensor:x
                                      withPaddingMode:mode
                                          leftPadding:lpad
                                         rightPadding:rpad
                                        constantValue:constant_value
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

/*** MIRROR PAD GRADIENT ***/

// The gradient of a mirror pad folds the reflected borders back onto the
// interior: an element that was copied to the border accumulates the gradient
// from both the copy and the original. REFLECT skips the edge row when
// mirroring, SYMMETRIC includes it, and that one-element offset is the whole
// difference between the two modes.
void MirrorPadGrad_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor grad, paddings;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, paddings.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  const int rank = static_cast<int>(grad_shape.size());
  std::vector<int64_t> flat;
  if (!ReadHostVector(paddings.get(), &flat, status)) return;
  if (static_cast<int>(flat.size()) != rank * 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MirrorPadGrad expects a [rank, 2] paddings tensor.");
    return;
  }

  std::vector<int64_t> left(rank), right(rank), out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    left[i] = flat[i * 2];
    right[i] = flat[i * 2 + 1];
    out_shape[i] = grad_shape[i] - left[i] - right[i];
    if (out_shape[i] < 1 || left[i] < 0 || right[i] < 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: MirrorPadGrad paddings do not fit the gradient.");
      return;
    }
  }
  // REFLECT mirrors about the edge without repeating it, so it cannot pad by
  // more than extent-1; SYMMETRIC repeats the edge and allows extent.
  const int64_t edge_offset = op->pad_mode == MPSGraphPaddingModeReflect ? 1 : 0;
  for (int i = 0; i < rank; ++i) {
    const int64_t limit = out_shape[i] - edge_offset;
    if (left[i] > limit || right[i] > limit) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: MirrorPadGrad padding exceeds what the mode allows.");
      return;
    }
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

  std::string key = "MirrorPadGrad";
  AppendShapeToKey(grad_shape, &key);
  AppendShapeToKey(left, &key);
  AppendShapeToKey(right, &key);
  key.append("/m").append(std::to_string(static_cast<int>(op->pad_mode)));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(grad_shape)
                                            dataType:mps_dtype
                                                name:nil];
        // Start from the interior, then fold each border back one axis at a
        // time. Reversing the border slice puts each element opposite the
        // interior position it was mirrored from.
        MPSGraphTensor* acc = dy;
        for (int i = 0; i < rank; ++i) {
          const NSUInteger axis = static_cast<NSUInteger>(i);
          MPSGraphTensor* interior =
              [g sliceTensor:acc
                   dimension:axis
                       start:static_cast<NSInteger>(left[i])
                      length:static_cast<NSInteger>(out_shape[i])
                        name:nil];
          if (left[i] > 0) {
            MPSGraphTensor* head = [g sliceTensor:acc
                                        dimension:axis
                                            start:0
                                           length:static_cast<NSInteger>(left[i])
                                             name:nil];
            MPSGraphTensor* folded =
                [g reverseTensor:head axes:@[ @(static_cast<NSInteger>(i)) ]
                            name:nil];
            // The fold lands at offset edge_offset from the low edge.
            std::vector<int64_t> pad_l(rank, 0), pad_r(rank, 0);
            pad_l[i] = edge_offset;
            pad_r[i] = out_shape[i] - left[i] - edge_offset;
            interior = [g additionWithPrimaryTensor:interior
                                    secondaryTensor:
                                        [g padTensor:folded
                                     withPaddingMode:MPSGraphPaddingModeConstant
                                         leftPadding:ToNSArray(pad_l)
                                        rightPadding:ToNSArray(pad_r)
                                       constantValue:0.0
                                                name:nil]
                                               name:nil];
          }
          if (right[i] > 0) {
            MPSGraphTensor* tail =
                [g sliceTensor:acc
                     dimension:axis
                         start:static_cast<NSInteger>(left[i] + out_shape[i])
                        length:static_cast<NSInteger>(right[i])
                          name:nil];
            MPSGraphTensor* folded =
                [g reverseTensor:tail axes:@[ @(static_cast<NSInteger>(i)) ]
                            name:nil];
            std::vector<int64_t> pad_l(rank, 0), pad_r(rank, 0);
            pad_l[i] = out_shape[i] - right[i] - edge_offset;
            pad_r[i] = edge_offset;
            interior = [g additionWithPrimaryTensor:interior
                                    secondaryTensor:
                                        [g padTensor:folded
                                     withPaddingMode:MPSGraphPaddingModeConstant
                                         leftPadding:ToNSArray(pad_l)
                                        rightPadding:ToNSArray(pad_r)
                                       constantValue:0.0
                                                name:nil]
                                               name:nil];
          }
          acc = interior;
        }
        [out->inputs addObject:dy];
        [out->outputs addObject:acc];
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

/*** REVERSE ***/

void ReverseV2_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input, axes_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, axes_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int rank = static_cast<int>(shape.size());
  std::vector<int64_t> axes;
  if (!ReadHostVector(axes_t.get(), &axes, status)) return;
  for (int64_t& a : axes) {
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Reverse axis is out of range.");
      return;
    }
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

  std::string key = "ReverseV2";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(axes, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  NSMutableArray<NSNumber*>* mps_axes = [NSMutableArray array];
  for (int64_t a : axes) [mps_axes addObject:@(static_cast<NSInteger>(a))];

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:(axes.empty()
                                     ? [out->graph identityWithTensor:x
                                                                 name:nil]
                                     : [out->graph reverseTensor:x
                                                            axes:mps_axes
                                                            name:nil])];
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

/*** SPLIT AND SPLITV ***/

// Split takes the axis first and divides evenly; SplitV takes explicit sizes
// and puts the axis last.
template <bool kVariable>
void Split_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor value, axis_t, sizes_t;
  if (kVariable) {
    TF_GetInput(ctx, 0, value.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_GetInput(ctx, 1, sizes_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_GetInput(ctx, 2, axis_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  } else {
    TF_GetInput(ctx, 0, axis_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_GetInput(ctx, 1, value.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> shape = ShapeOf(value.get());
  const int rank = static_cast<int>(shape.size());
  std::vector<int64_t> axis_vec;
  if (!ReadHostVector(axis_t.get(), &axis_vec, status)) return;
  if (axis_vec.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: Split has no axis.");
    return;
  }
  int64_t axis = axis_vec[0];
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Split axis is out of range.");
    return;
  }

  std::vector<int64_t> sizes;
  if (kVariable) {
    if (!ReadHostVector(sizes_t.get(), &sizes, status)) return;
    // A single -1 absorbs whatever the explicit sizes leave over.
    int negatives = 0;
    int64_t explicit_total = 0;
    for (int64_t s : sizes) {
      if (s < 0) negatives++;
      else explicit_total += s;
    }
    if (negatives > 1) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: SplitV allows at most one unknown size.");
      return;
    }
    for (int64_t& s : sizes) {
      if (s < 0) s = shape[axis] - explicit_total;
    }
  } else {
    const int n = op->num_split;
    if (n <= 0 || shape[axis] % n != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: Split requires the axis to divide evenly.");
      return;
    }
    sizes.assign(n, shape[axis] / n);
  }

  const int n = static_cast<int>(sizes.size());
  if (n != TF_NumOutputs(ctx)) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Split size count does not match the output count.");
    return;
  }

  std::vector<ScopedTensor> outputs(n);
  int64_t total = 0;
  for (int i = 0; i < n; ++i) {
    std::vector<int64_t> piece = shape;
    piece[axis] = sizes[i];
    total += sizes[i];
    outputs[i].reset(TF_AllocateOutput(
        ctx, i, op->dtype, piece.data(), rank,
        static_cast<size_t>(ElementCount(piece)) * TF_DataTypeSize(op->dtype),
        status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (total != shape[axis]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Split sizes do not sum to the axis length.");
    return;
  }
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Split";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(sizes, &key);
  key.append("/a").append(std::to_string(axis));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  NSMutableArray<NSNumber*>* split_sizes = [NSMutableArray array];
  for (int64_t s : sizes) [split_sizes addObject:@(static_cast<NSInteger>(s))];
  const NSInteger mps_axis = static_cast<NSInteger>(axis);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        NSArray<MPSGraphTensor*>* parts = [out->graph splitTensor:x
                                                       splitSizes:split_sizes
                                                             axis:mps_axis
                                                             name:nil];
        for (MPSGraphTensor* p in parts) [out->outputs addObject:p];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(value.get(), op->dtype, device, status);
  if (in_data == nil) return;
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  for (int i = 0; i < n; ++i) {
    MPSGraphTensorData* d =
        TensorDataForTensor(outputs[i].get(), op->dtype, device, status);
    if (d == nil) return;
    [results addObject:d];
  }
  RunGraph(stream, *cached, @[ in_data ], results, status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<DTypeOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(Slice_Compute, Slice_ComputeImpl)
METAL_COMPUTE(Pad_Compute, Pad_ComputeImpl<false>)
METAL_COMPUTE(PadV2_Compute, Pad_ComputeImpl<true>)
METAL_COMPUTE(ReverseV2_Compute, ReverseV2_ComputeImpl)
METAL_COMPUTE(MirrorPadGrad_Compute, MirrorPadGrad_ComputeImpl)
METAL_COMPUTE(Split_Compute, Split_ComputeImpl<false>)
METAL_COMPUTE(SplitV_Compute, Split_ComputeImpl<true>)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name,
              std::vector<const char*> host_args,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_INT32) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &DTypeOp_Create, compute, &DTypeOp_Delete);
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

void RegisterMetalSliceKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const std::string suffix = kSuffixes[i];
    for (int j = 0; j < 2; ++j) {
      const std::string isuffix = kIndexSuffixes[j];
      // Every shape argument is read on the host, so all of them stay off the
      // device.
      Register("Slice", &Slice_Compute, kDTypes[i],
               "MetalSlice" + suffix + isuffix, {"begin", "size"}, "Index",
               kIndexTypes[j]);
      Register("Pad", &Pad_Compute, kDTypes[i], "MetalPad" + suffix + isuffix,
               {"paddings"}, "Tpaddings", kIndexTypes[j]);
      Register("PadV2", &PadV2_Compute, kDTypes[i],
               "MetalPadV2" + suffix + isuffix, {"paddings"}, "Tpaddings",
               kIndexTypes[j]);
      Register("MirrorPad", &Pad_Compute, kDTypes[i],
               "MetalMirrorPad" + suffix + isuffix, {"paddings"}, "Tpaddings",
               kIndexTypes[j]);
      Register("MirrorPadGrad", &MirrorPadGrad_Compute, kDTypes[i],
               "MetalMirrorPadGrad" + suffix + isuffix, {"paddings"},
               "Tpaddings", kIndexTypes[j]);
      Register("ReverseV2", &ReverseV2_Compute, kDTypes[i],
               "MetalReverseV2" + suffix + isuffix, {"axis"}, "Tidx",
               kIndexTypes[j]);
      Register("SplitV", &SplitV_Compute, kDTypes[i],
               "MetalSplitV" + suffix + isuffix, {"size_splits", "split_dim"},
               "Tlen", kIndexTypes[j]);
    }
    Register("Split", &Split_Compute, kDTypes[i], "MetalSplit" + suffix,
             {"split_dim"});
  }
}

}  // namespace metal
}  // namespace tensorflow
