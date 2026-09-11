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

// SpaceToBatchND, BatchToSpaceND and their fixed-block SpaceToBatch and
// BatchToSpace forms, plus ReverseSequence.
//
// The block rearrangements are pure index permutations, so they are built out
// of pad, reshape and transpose rather than a shader. TensorFlow defines
// SpaceToBatchND as: pad each spatial axis, split it into [outer, block],
// then move every block axis in front of the batch. The reshape and transpose
// below are that definition written out; BatchToSpaceND is the same sequence
// reversed, ending in a crop.

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

struct BatchSpaceOp {
  TF_DataType dtype = TF_FLOAT;
  int64_t seq_dim = 0;
  int64_t batch_dim = 0;
  int64_t block_size = 2;
};

void* BatchSpaceOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BatchSpaceOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  int64_t v = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seq_dim", &v, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->seq_dim = v;
  v = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "batch_dim", &v, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->batch_dim = v;
  int32_t block = 2;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "block_size", &block, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->block_size = block;
  TF_DeleteStatus(status);
  return op;
}

void BatchSpaceOp_Delete(void* kernel) {
  delete static_cast<BatchSpaceOp*>(kernel);
}

NSArray<NSNumber*>* ToNS(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

/*** SPACE TO BATCH ND ***/

void SpaceToBatchND_ComputeImpl(BatchSpaceOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor input, block_t, pad_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, block_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, pad_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  std::vector<int64_t> block, pads;
  if (!ReadHostVector(block_t.get(), &block, status)) return;
  if (!ReadHostVector(pad_t.get(), &pads, status)) return;
  const int m = static_cast<int>(block.size());
  if (static_cast<int>(pads.size()) != 2 * m ||
      static_cast<int>(in_shape.size()) < m + 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SpaceToBatchND block and paddings are inconsistent "
                 "with the input rank.");
    return;
  }

  // Padded shape, then the split of each blocked axis into [outer, block].
  std::vector<int64_t> padded = in_shape;
  std::vector<int64_t> left(in_shape.size(), 0), right(in_shape.size(), 0);
  for (int i = 0; i < m; ++i) {
    left[1 + i] = pads[2 * i];
    right[1 + i] = pads[2 * i + 1];
    padded[1 + i] += pads[2 * i] + pads[2 * i + 1];
    if (block[i] <= 0 || padded[1 + i] % block[i] != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: SpaceToBatchND block size must divide the padded "
                   "extent.");
      return;
    }
  }

  // [batch, out_0, blk_0, out_1, blk_1, ..., remaining...]
  std::vector<int64_t> split;
  split.push_back(padded[0]);
  for (int i = 0; i < m; ++i) {
    split.push_back(padded[1 + i] / block[i]);
    split.push_back(block[i]);
  }
  for (size_t i = m + 1; i < padded.size(); ++i) split.push_back(padded[i]);

  // Block axes first, then batch, then the outer spatial axes and the rest.
  std::vector<int64_t> perm;
  for (int i = 0; i < m; ++i) perm.push_back(2 + 2 * i);
  perm.push_back(0);
  for (int i = 0; i < m; ++i) perm.push_back(1 + 2 * i);
  for (size_t i = 1 + 2 * m; i < split.size(); ++i) perm.push_back(i);

  std::vector<int64_t> out_shape;
  int64_t batch = padded[0];
  for (int i = 0; i < m; ++i) batch *= block[i];
  out_shape.push_back(batch);
  for (int i = 0; i < m; ++i) out_shape.push_back(padded[1 + i] / block[i]);
  for (size_t i = m + 1; i < padded.size(); ++i) out_shape.push_back(padded[i]);

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

  std::string key = "SpaceToBatchND";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(block, &key);
  AppendShapeToKey(pads, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* p = [g padTensor:x
                         withPaddingMode:MPSGraphPaddingModeConstant
                             leftPadding:ToNS(left)
                            rightPadding:ToNS(right)
                           constantValue:0.0
                                    name:nil];
        MPSGraphTensor* r = [g reshapeTensor:p
                                   withShape:MPSShape(split)
                                        name:nil];
        MPSGraphTensor* t = [g transposeTensor:r
                                   permutation:ToNS(perm)
                                          name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[g reshapeTensor:t
                                       withShape:MPSShape(out_shape)
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

/*** BATCH TO SPACE ND ***/

void BatchToSpaceND_ComputeImpl(BatchSpaceOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor input, block_t, crop_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, block_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, crop_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  std::vector<int64_t> block, crops;
  if (!ReadHostVector(block_t.get(), &block, status)) return;
  if (!ReadHostVector(crop_t.get(), &crops, status)) return;
  const int m = static_cast<int>(block.size());
  if (static_cast<int>(crops.size()) != 2 * m ||
      static_cast<int>(in_shape.size()) < m + 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BatchToSpaceND block and crops are inconsistent with "
                 "the input rank.");
    return;
  }
  int64_t block_total = 1;
  for (int64_t b : block) {
    if (b <= 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: BatchToSpaceND block sizes must be positive.");
      return;
    }
    block_total *= b;
  }
  if (in_shape[0] % block_total != 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BatchToSpaceND batch must divide by the block "
                 "product.");
    return;
  }

  // [blk_0, ..., blk_{m-1}, batch, out_0, ..., rest...]
  std::vector<int64_t> split;
  for (int i = 0; i < m; ++i) split.push_back(block[i]);
  split.push_back(in_shape[0] / block_total);
  for (int i = 0; i < m; ++i) split.push_back(in_shape[1 + i]);
  for (size_t i = m + 1; i < in_shape.size(); ++i) split.push_back(in_shape[i]);

  // Undo the SpaceToBatch permutation: batch first, then each outer axis
  // followed by its block axis.
  std::vector<int64_t> perm;
  perm.push_back(m);
  for (int i = 0; i < m; ++i) {
    perm.push_back(m + 1 + i);
    perm.push_back(i);
  }
  for (size_t i = 2 * m + 1; i < split.size(); ++i) perm.push_back(i);

  std::vector<int64_t> merged;
  merged.push_back(in_shape[0] / block_total);
  for (int i = 0; i < m; ++i) merged.push_back(in_shape[1 + i] * block[i]);
  for (size_t i = m + 1; i < in_shape.size(); ++i) merged.push_back(in_shape[i]);

  std::vector<int64_t> out_shape = merged;
  for (int i = 0; i < m; ++i) {
    out_shape[1 + i] -= crops[2 * i] + crops[2 * i + 1];
    if (out_shape[1 + i] < 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: BatchToSpaceND crops exceed the extent.");
      return;
    }
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

  std::string key = "BatchToSpaceND";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(block, &key);
  AppendShapeToKey(crops, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* r = [g reshapeTensor:x
                                   withShape:MPSShape(split)
                                        name:nil];
        MPSGraphTensor* t = [g transposeTensor:r
                                   permutation:ToNS(perm)
                                          name:nil];
        MPSGraphTensor* merged_t = [g reshapeTensor:t
                                          withShape:MPSShape(merged)
                                               name:nil];
        // The crop is the trailing slice, one axis at a time.
        for (int i = 0; i < m; ++i) {
          if (crops[2 * i] == 0 && crops[2 * i + 1] == 0) continue;
          merged_t = [g sliceTensor:merged_t
                          dimension:static_cast<NSUInteger>(1 + i)
                              start:static_cast<NSInteger>(crops[2 * i])
                             length:static_cast<NSInteger>(out_shape[1 + i])
                               name:nil];
        }
        [out->inputs addObject:x];
        [out->outputs addObject:merged_t];
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

/*** FIXED-BLOCK SPACE TO BATCH AND BACK ***/

// SpaceToBatch and BatchToSpace are the pre-ND spellings: the block size is a
// scalar attribute and the layout is fixed at NHWC, so they reduce to the ND
// forms with a two-element block shape. The pad or crop tensor keeps the same
// [2, 2] layout, so it is read the same way.
template <bool kToBatch>
void FixedBlock_ComputeImpl(BatchSpaceOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input, edge_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, edge_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  if (op->block_size < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: block_size must be at least 2.");
    return;
  }
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SpaceToBatch expects a rank-4 NHWC input.");
    return;
  }
  std::vector<int64_t> edges;
  if (!ReadHostVector(edge_t.get(), &edges, status)) return;
  if (edges.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the paddings or crops tensor must be [2, 2].");
    return;
  }

  const int64_t b = op->block_size;
  std::vector<int64_t> out_shape(4);
  if (kToBatch) {
    const int64_t ph = in_shape[1] + edges[0] + edges[1];
    const int64_t pw = in_shape[2] + edges[2] + edges[3];
    if (ph % b != 0 || pw % b != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: block_size must divide the padded extents.");
      return;
    }
    out_shape = {in_shape[0] * b * b, ph / b, pw / b, in_shape[3]};
  } else {
    if (in_shape[0] % (b * b) != 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: batch must divide by block_size squared.");
      return;
    }
    const int64_t h = in_shape[1] * b - edges[0] - edges[1];
    const int64_t w = in_shape[2] * b - edges[2] - edges[3];
    if (h < 0 || w < 0) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: crops exceed the extent.");
      return;
    }
    out_shape = {in_shape[0] / (b * b), h, w, in_shape[3]};
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = kToBatch ? "SpaceToBatch" : "BatchToSpace";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(edges, &key);
  key.append("/b").append(std::to_string(b));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* r;
        if (kToBatch) {
          std::vector<int64_t> left = {0, edges[0], edges[2], 0};
          std::vector<int64_t> right = {0, edges[1], edges[3], 0};
          MPSGraphTensor* p = [g padTensor:x
                           withPaddingMode:MPSGraphPaddingModeConstant
                               leftPadding:ToNS(left)
                              rightPadding:ToNS(right)
                             constantValue:0.0
                                      name:nil];
          const int64_t ph = in_shape[1] + edges[0] + edges[1];
          const int64_t pw = in_shape[2] + edges[2] + edges[3];
          std::vector<int64_t> split = {in_shape[0], ph / b, b,
                                        pw / b,      b,     in_shape[3]};
          MPSGraphTensor* rs = [g reshapeTensor:p
                                      withShape:MPSShape(split)
                                           name:nil];
          // Block axes to the front, then batch, then the outer extents.
          MPSGraphTensor* tr = [g transposeTensor:rs
                                      permutation:@[ @2, @4, @0, @1, @3, @5 ]
                                             name:nil];
          r = [g reshapeTensor:tr withShape:MPSShape(out_shape) name:nil];
        } else {
          std::vector<int64_t> split = {b,
                                        b,
                                        in_shape[0] / (b * b),
                                        in_shape[1],
                                        in_shape[2],
                                        in_shape[3]};
          MPSGraphTensor* rs = [g reshapeTensor:x
                                      withShape:MPSShape(split)
                                           name:nil];
          MPSGraphTensor* tr = [g transposeTensor:rs
                                      permutation:@[ @2, @3, @0, @4, @1, @5 ]
                                             name:nil];
          std::vector<int64_t> merged = {in_shape[0] / (b * b), in_shape[1] * b,
                                         in_shape[2] * b, in_shape[3]};
          MPSGraphTensor* m = [g reshapeTensor:tr
                                     withShape:MPSShape(merged)
                                          name:nil];
          if (edges[0] != 0 || edges[1] != 0) {
            m = [g sliceTensor:m
                     dimension:1
                         start:static_cast<NSInteger>(edges[0])
                        length:static_cast<NSInteger>(out_shape[1])
                          name:nil];
          }
          if (edges[2] != 0 || edges[3] != 0) {
            m = [g sliceTensor:m
                     dimension:2
                         start:static_cast<NSInteger>(edges[2])
                        length:static_cast<NSInteger>(out_shape[2])
                          name:nil];
          }
          r = m;
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

/*** REVERSE SEQUENCE ***/

// ReverseSequence reverses the first seq_lengths[i] entries of row i along
// seq_dim, leaving the tail in place. Rows have different lengths, so this
// cannot be one reverse: it is a full reverse combined with the original
// under a per-position mask derived from the coordinate along seq_dim.
void ReverseSequence_ComputeImpl(BatchSpaceOp* op, TF_OpKernelContext* ctx,
                                 TF_Status* status) {
  ScopedTensor input, lengths;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, lengths.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int rank = static_cast<int>(shape.size());
  int64_t seq_dim = op->seq_dim;
  int64_t batch_dim = op->batch_dim;
  if (seq_dim < 0) seq_dim += rank;
  if (batch_dim < 0) batch_dim += rank;
  if (seq_dim < 0 || seq_dim >= rank || batch_dim < 0 || batch_dim >= rank ||
      seq_dim == batch_dim) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ReverseSequence seq_dim and batch_dim are invalid.");
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
  MPSDataType len_dtype;
  if (!MPSTypeFor(TF_TensorType(lengths.get()), &len_dtype, status)) return;

  // The lengths vector is per batch entry; it broadcasts against the data once
  // reshaped to sit on batch_dim.
  std::vector<int64_t> len_shape(rank, 1);
  len_shape[batch_dim] = shape[batch_dim];

  std::string key = "ReverseSequence";
  AppendShapeToKey(shape, &key);
  key.append("/s").append(std::to_string(seq_dim));
  key.append("/b").append(std::to_string(batch_dim));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSInteger sd = static_cast<NSInteger>(seq_dim);
  const int64_t seq_len = shape[seq_dim];

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* len = [g placeholderWithShape:MPSShape(len_shape)
                                             dataType:len_dtype
                                                 name:nil];
        MPSGraphTensor* len_i = [g castTensor:len
                                       toType:MPSDataTypeInt32
                                         name:nil];
        MPSGraphTensor* pos = [g coordinateAlongAxis:sd
                                           withShape:MPSShape(shape)
                                                name:nil];
        MPSGraphTensor* pos_i = [g castTensor:pos
                                       toType:MPSDataTypeInt32
                                         name:nil];
        // Reversing the whole axis puts entry p at seq_len-1-p; the entry that
        // belongs at p within a row of length L is L-1-p, so the fully
        // reversed tensor has to be rolled by seq_len-L. Building that with a
        // gather would need per-row indices, which MPSGraph cannot express
        // here, so the two extremes are combined instead: positions inside the
        // sequence take the reversed value shifted by the row's own offset,
        // which is exactly a gather along seq_dim.
        MPSGraphTensor* limit =
            [g lessThanWithPrimaryTensor:pos_i secondaryTensor:len_i name:nil];
        MPSGraphTensor* one =
            [g constantWithScalar:1.0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* source_index =
            [g subtractionWithPrimaryTensor:
                   [g subtractionWithPrimaryTensor:len_i
                                   secondaryTensor:one
                                              name:nil]
                            secondaryTensor:pos_i
                                       name:nil];
        // Outside the sequence the position maps to itself.
        MPSGraphTensor* index =
            [g selectWithPredicateTensor:limit
                     truePredicateTensor:source_index
                    falsePredicateTensor:pos_i
                                    name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:len];
        [out->outputs addObject:[g gatherAlongAxis:sd
                                 withUpdatesTensor:x
                                     indicesTensor:index
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice len_slice;
  if (!SliceForTensor(lengths.get(), &len_slice, status)) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* l_data =
      TensorDataFor(len_slice, len_shape, TF_TensorType(lengths.get()), device,
                    status);
  if (l_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, l_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<BatchSpaceOp*>(kernel);                            \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(SpaceToBatchND_Compute, SpaceToBatchND_ComputeImpl)
METAL_COMPUTE(BatchToSpaceND_Compute, BatchToSpaceND_ComputeImpl)
METAL_COMPUTE(ReverseSequence_Compute, ReverseSequence_ComputeImpl)
METAL_COMPUTE(SpaceToBatch_Compute, FixedBlock_ComputeImpl<true>)
METAL_COMPUTE(BatchToSpace_Compute, FixedBlock_ComputeImpl<false>)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &BatchSpaceOp_Create,
                          compute, &BatchSpaceOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalBatchSpaceKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    // The block shape and the pad or crop amounts are read on the host to
    // build the reshape and permutation.
    Register("SpaceToBatchND", &SpaceToBatchND_Compute, t,
             "MetalSpaceToBatchND" + s, {"block_shape", "paddings"});
    Register("BatchToSpaceND", &BatchToSpaceND_Compute, t,
             "MetalBatchToSpaceND" + s, {"block_shape", "crops"});
    // The sequence lengths stay on the device; only seq_dim and batch_dim,
    // which are attributes, are needed on the host.
    Register("ReverseSequence", &ReverseSequence_Compute, t,
             "MetalReverseSequence" + s, {});
    Register("SpaceToBatch", &SpaceToBatch_Compute, t, "MetalSpaceToBatch" + s,
             {"paddings"});
    Register("BatchToSpace", &BatchToSpace_Compute, t, "MetalBatchToSpace" + s,
             {"crops"});
  }
}

}  // namespace metal
}  // namespace tensorflow
