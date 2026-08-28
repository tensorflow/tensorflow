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

// Gather, OneHot, TopK, the cumulative scans, MatrixBandPart, the space and
// depth rearrangements, ClipByValue and the norm reductions.

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

struct IndexOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType index_dtype = TF_INT32;
  int batch_dims = 0;
  bool exclusive = false;
  bool reverse = false;
  bool sorted = true;
  bool keep_dims = false;
  int block_size = 2;
  float tolerance = 1e-5f;
  int one_hot_axis = -1;
};

void* IndexOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new IndexOp();
  // GatherV2 names its element type Tparams; the other ops here call it T.
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    TF_OpKernelConstruction_GetAttrType(ctx, "Tparams", &op->dtype, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  // Everything below is optional and differs per op; a missing attribute keeps
  // the default rather than failing the whole construction.
  int32_t batch_dims = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "batch_dims", &batch_dims, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->batch_dims = batch_dims;

  int32_t block_size = 2;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "block_size", &block_size, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->block_size = block_size;

  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "exclusive", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->exclusive = flag != 0;
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "reverse", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->reverse = flag != 0;
  flag = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "sorted", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->sorted = flag != 0;
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "keep_dims", &flag, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->keep_dims = flag != 0;

  TF_OpKernelConstruction_GetAttrFloat(ctx, "tolerance", &op->tolerance, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");

  TF_OpKernelConstruction_GetAttrType(ctx, "Tindices", &op->index_dtype, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");

  int32_t one_hot_axis = -1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "axis", &one_hot_axis, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->one_hot_axis = one_hot_axis;

  TF_DeleteStatus(status);
  return op;
}

void IndexOp_Delete(void* kernel) { delete static_cast<IndexOp*>(kernel); }

/*** GATHER ***/

void GatherV2_ComputeImpl(IndexOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor params, indices, axis_t;
  TF_GetInput(ctx, 0, params.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> p_shape = ShapeOf(params.get());
  const std::vector<int64_t> i_shape = ShapeOf(indices.get());
  const int rank = static_cast<int>(p_shape.size());

  int64_t axis = 0;
  if (TF_NumInputs(ctx) > 2) {
    TF_GetInput(ctx, 2, axis_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    std::vector<int64_t> v;
    if (!ReadHostVector(axis_t.get(), &v, status)) return;
    if (!v.empty()) axis = v[0];
  }
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Gather axis is out of range.");
    return;
  }

  // Result: params up to axis, then the index shape past the batch dims, then
  // params after axis.
  const int bd = op->batch_dims;
  std::vector<int64_t> out_shape;
  for (int i = 0; i < axis; ++i) out_shape.push_back(p_shape[i]);
  for (size_t i = bd; i < i_shape.size(); ++i) out_shape.push_back(i_shape[i]);
  for (int i = axis + 1; i < rank; ++i) out_shape.push_back(p_shape[i]);

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

  MPSDataType mps_dtype, idx_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(TF_TensorType(indices.get()), &idx_dtype, status)) return;

  std::string key = "GatherV2";
  AppendShapeToKey(p_shape, &key);
  AppendShapeToKey(i_shape, &key);
  key.append("/a").append(std::to_string(axis));
  key.append("/b").append(std::to_string(bd));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger mps_axis = static_cast<NSUInteger>(axis);
  const NSUInteger mps_bd = static_cast<NSUInteger>(bd);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* p = [out->graph placeholderWithShape:MPSShape(p_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* i = [out->graph placeholderWithShape:MPSShape(i_shape)
                                                    dataType:idx_dtype
                                                        name:nil];
        [out->inputs addObject:p];
        [out->inputs addObject:i];
        [out->outputs addObject:[out->graph gatherWithUpdatesTensor:p
                                                      indicesTensor:i
                                                               axis:mps_axis
                                                    batchDimensions:mps_bd
                                                               name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* p_data =
      TensorDataForTensor(params.get(), op->dtype, device, status);
  if (p_data == nil) return;
  MPSGraphTensorData* i_data = TensorDataForTensor(
      indices.get(), TF_TensorType(indices.get()), device, status);
  if (i_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ p_data, i_data ], @[ o_data ], status);
}

/*** ONE HOT ***/

void OneHot_ComputeImpl(IndexOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor indices, depth_t, on_t, off_t;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, depth_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, on_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, off_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> depth_v;
  if (!ReadHostVector(depth_t.get(), &depth_v, status)) return;
  if (depth_v.empty() || depth_v[0] < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: OneHot depth is invalid.");
    return;
  }
  const int64_t depth = depth_v[0];

  const std::vector<int64_t> i_shape = ShapeOf(indices.get());
  // TensorFlow's default axis of -1 appends the new dimension; any other value
  // inserts it there.
  int64_t axis = op->one_hot_axis;
  if (axis < 0) axis += static_cast<int64_t>(i_shape.size()) + 1;
  if (axis < 0 || axis > static_cast<int64_t>(i_shape.size())) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: OneHot axis is out of range.");
    return;
  }
  std::vector<int64_t> out_shape = i_shape;
  out_shape.insert(out_shape.begin() + axis, depth);
  const NSUInteger mps_axis = static_cast<NSUInteger>(axis);

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

  MPSDataType mps_dtype, idx_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(TF_TensorType(indices.get()), &idx_dtype, status)) return;

  std::string key = "OneHot";
  AppendShapeToKey(i_shape, &key);
  key.append("/d").append(std::to_string(depth));
  key.append("/x").append(std::to_string(axis));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const std::vector<int64_t> scalar_shape = {1};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* idx = [g placeholderWithShape:MPSShape(i_shape)
                                             dataType:idx_dtype
                                                 name:nil];
        MPSGraphTensor* on = [g placeholderWithShape:MPSShape(scalar_shape)
                                            dataType:mps_dtype
                                                name:nil];
        MPSGraphTensor* off = [g placeholderWithShape:MPSShape(scalar_shape)
                                             dataType:mps_dtype
                                                 name:nil];
        MPSGraphTensor* i32 =
            [g castTensor:idx toType:MPSDataTypeInt32 name:nil];
        // Built with 1 and 0 and then rescaled, rather than passing on_value
        // and off_value to MPSGraph, which takes them as compile-time doubles.
        // Reading two device scalars on the host would drain the stream on
        // every call; this keeps everything on device.
        MPSGraphTensor* hot =
            [g oneHotWithIndicesTensor:i32
                                 depth:static_cast<NSUInteger>(depth)
                                  axis:mps_axis
                              dataType:mps_dtype
                               onValue:1.0
                              offValue:0.0
                                  name:nil];
        MPSGraphTensor* span =
            [g subtractionWithPrimaryTensor:on secondaryTensor:off name:nil];
        MPSGraphTensor* scaled =
            [g multiplicationWithPrimaryTensor:hot
                               secondaryTensor:span
                                          name:nil];
        [out->inputs addObject:idx];
        [out->inputs addObject:on];
        [out->inputs addObject:off];
        [out->outputs addObject:[g additionWithPrimaryTensor:scaled
                                             secondaryTensor:off
                                                        name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice on_slice, off_slice;
  if (!SliceForTensor(on_t.get(), &on_slice, status)) return;
  if (!SliceForTensor(off_t.get(), &off_slice, status)) return;

  MPSGraphTensorData* i_data = TensorDataForTensor(
      indices.get(), TF_TensorType(indices.get()), device, status);
  if (i_data == nil) return;
  MPSGraphTensorData* on_data =
      TensorDataFor(on_slice, scalar_shape, op->dtype, device, status);
  if (on_data == nil) return;
  MPSGraphTensorData* off_data =
      TensorDataFor(off_slice, scalar_shape, op->dtype, device, status);
  if (off_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ i_data, on_data, off_data ], @[ o_data ],
           status);
}

/*** TOP K ***/

void TopKV2_ComputeImpl(IndexOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor input, k_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, k_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: TopK needs a tensor of rank at least 1.");
    return;
  }
  std::vector<int64_t> k_v;
  if (!ReadHostVector(k_t.get(), &k_v, status)) return;
  if (k_v.empty() || k_v[0] < 0 || k_v[0] > in_shape.back()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: TopK k is out of range.");
    return;
  }
  const int64_t k = k_v[0];

  std::vector<int64_t> out_shape = in_shape;
  out_shape.back() = k;
  const int64_t count = ElementCount(out_shape);

  ScopedTensor values, idx_out;
  values.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  idx_out.reset(TF_AllocateOutput(
      ctx, 1, TF_INT32, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "TopKV2";
  AppendShapeToKey(in_shape, &key);
  key.append("/k").append(std::to_string(k));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        NSArray<MPSGraphTensor*>* r =
            [out->graph topKWithSourceTensor:x
                                            k:static_cast<NSUInteger>(k)
                                         name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:r[0]];
        // MPSGraph returns int32 indices already, which is what TopKV2
        // declares.
        [out->outputs addObject:[out->graph castTensor:r[1]
                                                toType:MPSDataTypeInt32
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(values.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* i_data =
      TensorDataForTensor(idx_out.get(), TF_INT32, device, status);
  if (i_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ v_data, i_data ], status);
}

/*** CUMULATIVE SCANS ***/

template <bool kProduct>
void Cumulative_ComputeImpl(IndexOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input, axis_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, axis_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int rank = static_cast<int>(shape.size());
  std::vector<int64_t> axis_v;
  if (!ReadHostVector(axis_t.get(), &axis_v, status)) return;
  int64_t axis = axis_v.empty() ? 0 : axis_v[0];
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: cumulative axis is out of range.");
    return;
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

  std::string key = kProduct ? "Cumprod" : "Cumsum";
  AppendShapeToKey(shape, &key);
  key.append("/a").append(std::to_string(axis));
  key.append(op->exclusive ? "/excl" : "/incl");
  key.append(op->reverse ? "/rev" : "/fwd");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSInteger mps_axis = static_cast<NSInteger>(axis);
  const BOOL exclusive = op->exclusive ? YES : NO;
  const BOOL reverse = op->reverse ? YES : NO;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs
            addObject:(kProduct
                           ? [out->graph cumulativeProductWithTensor:x
                                                                axis:mps_axis
                                                           exclusive:exclusive
                                                             reverse:reverse
                                                                name:nil]
                           : [out->graph cumulativeSumWithTensor:x
                                                            axis:mps_axis
                                                       exclusive:exclusive
                                                         reverse:reverse
                                                            name:nil])];
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

/*** CLIP BY VALUE ***/

void ClipByValue_ComputeImpl(IndexOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor t, lo, hi;
  TF_GetInput(ctx, 0, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, lo.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, hi.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(t.get());
  const std::vector<int64_t> lo_shape = ShapeOf(lo.get());
  const std::vector<int64_t> hi_shape = ShapeOf(hi.get());
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

  std::string key = "ClipByValue";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(lo_shape, &key);
  AppendShapeToKey(hi_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* a = [out->graph placeholderWithShape:MPSShape(lo_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* b = [out->graph placeholderWithShape:MPSShape(hi_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs addObject:[out->graph clampWithTensor:x
                                             minValueTensor:a
                                             maxValueTensor:b
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(t.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* a_data =
      TensorDataForTensor(lo.get(), op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataForTensor(hi.get(), op->dtype, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, a_data, b_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<IndexOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(GatherV2_Compute, GatherV2_ComputeImpl)
METAL_COMPUTE(OneHot_Compute, OneHot_ComputeImpl)
METAL_COMPUTE(TopKV2_Compute, TopKV2_ComputeImpl)
METAL_COMPUTE(Cumsum_Compute, Cumulative_ComputeImpl<false>)
METAL_COMPUTE(Cumprod_Compute, Cumulative_ComputeImpl<true>)
METAL_COMPUTE(ClipByValue_Compute, ClipByValue_ComputeImpl)

#undef METAL_COMPUTE

// `type_attr` is the name the op gives its element type. Most call it T;
// GatherV2 calls it Tparams, and constraining T there is a registration that
// can never match, which TensorFlow reports as "OpKernel 'GatherV2' has
// constraint on attr 'T' not in NodeDef".
void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_INT32,
              const char* type_attr = "T") {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &IndexOp_Create, compute, &IndexOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, type_attr, dtype, status);
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

void RegisterMetalIndexKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const std::string suffix = kSuffixes[i];
    Register("ClipByValue", &ClipByValue_Compute, kDTypes[i],
             "MetalClipByValue" + suffix, {});
    for (int j = 0; j < 2; ++j) {
      const std::string isuffix = kIndexSuffixes[j];
      // The gather axis, the one-hot depth, k, and the scan axis are all read
      // on the host to size the output.
      Register("GatherV2", &GatherV2_Compute, kDTypes[i],
               "MetalGatherV2" + suffix + isuffix, {"axis"}, "Tindices",
               kIndexTypes[j], "Tparams");
      Register("OneHot", &OneHot_Compute, kDTypes[i],
               "MetalOneHot" + suffix + isuffix, {"depth"}, "TI",
               kIndexTypes[j]);
      Register("Cumsum", &Cumsum_Compute, kDTypes[i],
               "MetalCumsum" + suffix + isuffix, {"axis"}, "Tidx",
               kIndexTypes[j]);
      Register("Cumprod", &Cumprod_Compute, kDTypes[i],
               "MetalCumprod" + suffix + isuffix, {"axis"}, "Tidx",
               kIndexTypes[j]);
    }
    Register("TopKV2", &TopKV2_Compute, kDTypes[i], "MetalTopKV2" + suffix,
             {"k"});
  }
}

}  // namespace metal
}  // namespace tensorflow
