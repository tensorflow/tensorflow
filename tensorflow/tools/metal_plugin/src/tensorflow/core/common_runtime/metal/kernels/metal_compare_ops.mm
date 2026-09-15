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

// Comparisons, logical operators, Select and ArgMax/ArgMin.
//
// These are what a metric is made of. `model.fit(..., metrics=["accuracy"])`
// emits ArgMax, Equal, Cast and Mean every step, so without them the accuracy
// computation copies the logits back to the host on every batch, which on a
// large output layer costs more than the step it is measuring.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

bool BroadcastShape(const std::vector<int64_t>& lhs,
                    const std::vector<int64_t>& rhs, std::vector<int64_t>* out,
                    TF_Status* status) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  out->assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t a = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const int64_t b = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (a != b && a != 1 && b != 1) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: operand shapes do not broadcast.");
      return false;
    }
    (*out)[i] = std::max(a, b);
  }
  return true;
}

/*** COMPARISONS ***/

enum class CompareKind { kEqual, kNotEqual, kLess, kLessEqual, kGreater,
                         kGreaterEqual, kLogicalAnd, kLogicalOr,
                         kApproximateEqual };

const char* NameOf(CompareKind k) {
  switch (k) {
    case CompareKind::kEqual: return "Equal";
    case CompareKind::kNotEqual: return "NotEqual";
    case CompareKind::kLess: return "Less";
    case CompareKind::kLessEqual: return "LessEqual";
    case CompareKind::kGreater: return "Greater";
    case CompareKind::kGreaterEqual: return "GreaterEqual";
    case CompareKind::kLogicalAnd: return "LogicalAnd";
    case CompareKind::kLogicalOr: return "LogicalOr";
    case CompareKind::kApproximateEqual: return "ApproximateEqual";
  }
  return "?";
}

MPSGraphTensor* ApplyCompare(MPSGraph* g, CompareKind k, MPSGraphTensor* a,
                             MPSGraphTensor* b, float tolerance,
                             MPSDataType dtype) {
  switch (k) {
    case CompareKind::kEqual:
      return [g equalWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kNotEqual:
      return [g notEqualWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kLess:
      return [g lessThanWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kLessEqual:
      return [g lessThanOrEqualToWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kGreater:
      return [g greaterThanWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kGreaterEqual:
      return [g greaterThanOrEqualToWithPrimaryTensor:a
                                      secondaryTensor:b
                                                 name:nil];
    case CompareKind::kLogicalAnd:
      return [g logicalANDWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kLogicalOr:
      return [g logicalORWithPrimaryTensor:a secondaryTensor:b name:nil];
    case CompareKind::kApproximateEqual: {
      // |a - b| < tolerance, strictly, which is how TensorFlow defines it.
      MPSGraphTensor* d = [g absoluteWithTensor:
                                 [g subtractionWithPrimaryTensor:a
                                                 secondaryTensor:b
                                                            name:nil]
                                           name:nil];
      return [g lessThanWithPrimaryTensor:d
                          secondaryTensor:[g constantWithScalar:tolerance
                                                       dataType:dtype]
                                     name:nil];
    }
  }
  return nil;
}

struct DTypeOp {
  TF_DataType dtype = TF_FLOAT;
  float tolerance = 1e-5f;  // ApproximateEqual's default
};

void* DTypeOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DTypeOp();
  // The logical operators are bool on both sides and therefore have no T
  // attribute at all. Insisting on one made LogicalAnd, LogicalOr and
  // LogicalNot fail construction with "No attr named 'T' in NodeDef", so all
  // three were registered and unusable.
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_BOOL;
  }
  TF_OpKernelConstruction_GetAttrFloat(ctx, "tolerance", &op->tolerance,
                                       status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void DTypeOp_Delete(void* kernel) { delete static_cast<DTypeOp*>(kernel); }

template <CompareKind kKind>
void Compare_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor lhs, rhs;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> lhs_shape = ShapeOf(lhs.get());
  const std::vector<int64_t> rhs_shape = ShapeOf(rhs.get());
  std::vector<int64_t> out_shape;
  if (!BroadcastShape(lhs_shape, rhs_shape, &out_shape, status)) return;

  // Comparisons always produce bool, whatever the operands were.
  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_BOOL, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_BOOL), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = NameOf(kKind);
  AppendShapeToKey(lhs_shape, &key);
  AppendShapeToKey(rhs_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/e").append(std::to_string(op->tolerance));
  const float tolerance = op->tolerance;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* a = [out->graph placeholderWithShape:MPSShape(lhs_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* b = [out->graph placeholderWithShape:MPSShape(rhs_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* r =
            ApplyCompare(out->graph, kKind, a, b, tolerance, mps_dtype);
        // MPSGraph comparisons yield the operand type, so the result is cast
        // to bool to match what TensorFlow declared the output to be.
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs addObject:[out->graph castTensor:r
                                                toType:MPSDataTypeBool
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* a_data =
      TensorDataForTensor(lhs.get(), op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataForTensor(rhs.get(), op->dtype, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), TF_BOOL, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ a_data, b_data ], @[ out_data ], status);
}

/*** LOGICAL NOT ***/

void LogicalNot_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_BOOL, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_BOOL), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "LogicalNot";
  AppendShapeToKey(shape, &key);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:MPSDataTypeBool
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph notWithTensor:x name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), TF_BOOL, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), TF_BOOL, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** SELECT ***/

void Select_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor cond, a, b;
  TF_GetInput(ctx, 0, cond.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, a.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, b.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> cond_shape = ShapeOf(cond.get());
  const std::vector<int64_t> a_shape = ShapeOf(a.get());
  const std::vector<int64_t> b_shape = ShapeOf(b.get());
  std::vector<int64_t> ab_shape, out_shape;
  if (!BroadcastShape(a_shape, b_shape, &ab_shape, status)) return;
  if (!BroadcastShape(cond_shape, ab_shape, &out_shape, status)) return;

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

  std::string key = "Select";
  AppendShapeToKey(cond_shape, &key);
  AppendShapeToKey(a_shape, &key);
  AppendShapeToKey(b_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* c = [out->graph placeholderWithShape:MPSShape(cond_shape)
                                                    dataType:MPSDataTypeBool
                                                        name:nil];
        MPSGraphTensor* t = [out->graph placeholderWithShape:MPSShape(a_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* f = [out->graph placeholderWithShape:MPSShape(b_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:c];
        [out->inputs addObject:t];
        [out->inputs addObject:f];
        [out->outputs addObject:[out->graph selectWithPredicateTensor:c
                                                  truePredicateTensor:t
                                                 falsePredicateTensor:f
                                                                 name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* c_data =
      TensorDataForTensor(cond.get(), TF_BOOL, device, status);
  if (c_data == nil) return;
  MPSGraphTensorData* t_data =
      TensorDataForTensor(a.get(), op->dtype, device, status);
  if (t_data == nil) return;
  MPSGraphTensorData* f_data =
      TensorDataForTensor(b.get(), op->dtype, device, status);
  if (f_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ c_data, t_data, f_data ], @[ out_data ], status);
}

/*** ARG MAX AND ARG MIN ***/

struct ArgOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType out_dtype = TF_INT64;
};

void* ArgOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ArgOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "output_type", &op->out_dtype,
                                        status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      op->out_dtype = TF_INT64;
    }
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void ArgOp_Delete(void* kernel) { delete static_cast<ArgOp*>(kernel); }

template <bool kMaximum>
void Arg_ComputeImpl(ArgOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input, dim_tensor;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, dim_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  if (rank == 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ArgMax needs a tensor of rank at least 1.");
    return;
  }

  // The dimension arrives in host memory, so it can shape the output directly.
  const void* dim_data = TF_TensorData(dim_tensor.get());
  int64_t axis = TF_TensorType(dim_tensor.get()) == TF_INT32
                     ? *static_cast<const int32_t*>(dim_data)
                     : *static_cast<const int64_t*>(dim_data);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ArgMax dimension is out of range.");
    return;
  }

  // TensorFlow drops the reduced axis; MPSGraph keeps it at extent 1.
  std::vector<int64_t> out_shape;
  for (int i = 0; i < rank; ++i) {
    if (i != axis) out_shape.push_back(in_shape[i]);
  }
  std::vector<int64_t> kept_shape = in_shape;
  kept_shape[axis] = 1;

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->out_dtype, out_shape.data(),
      static_cast<int>(out_shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->out_dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype, out_mps;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  if (!MPSTypeFor(op->out_dtype, &out_mps, status)) return;

  std::string key = kMaximum ? "ArgMax" : "ArgMin";
  AppendShapeToKey(in_shape, &key);
  key.append("/a").append(std::to_string(axis));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/o").append(std::to_string(static_cast<int>(op->out_dtype)));
  const NSInteger mps_axis = static_cast<NSInteger>(axis);
  const std::vector<int64_t> flat = out_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* r =
            kMaximum ? [out->graph reductionArgMaximumWithTensor:x
                                                            axis:mps_axis
                                                            name:nil]
                     : [out->graph reductionArgMinimumWithTensor:x
                                                            axis:mps_axis
                                                            name:nil];
        MPSGraphTensor* cast = [out->graph castTensor:r
                                               toType:out_mps
                                                 name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph reshapeTensor:cast
                                                withShape:MPSShape(flat)
                                                     name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->out_dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** IN TOP K ***/

// InTopK asks, per row, whether the target class is among the k highest
// predictions. Rather than sorting, it counts how many entries strictly beat
// the target: the target is in the top k exactly when fewer than k do. That
// also gives TensorFlow's tie behaviour for free, since ties do not count as
// beating.
struct TopKOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType index_dtype = TF_INT32;
  int64_t k = 1;
  bool k_from_attr = false;
};

void* TopKOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new TopKOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->index_dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->index_dtype = TF_INT32;
  }
  int64_t k = 1;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "k", &k, status);
  if (TF_GetCode(status) == TF_OK) {
    op->k = k;
    op->k_from_attr = true;
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
  TF_DeleteStatus(status);
  return op;
}

void TopKOp_Delete(void* kernel) { delete static_cast<TopKOp*>(kernel); }

void InTopK_ComputeImpl(TopKOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor predictions, targets;
  TF_GetInput(ctx, 0, predictions.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, targets.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  int64_t k = op->k;
  if (!op->k_from_attr) {
    ScopedTensor k_t;
    TF_GetInput(ctx, 2, k_t.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    const void* p = TF_TensorData(k_t.get());
    if (p == nullptr) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "Metal: InTopKV2 k has no data.");
      return;
    }
    k = TF_TensorType(k_t.get()) == TF_INT32
            ? *static_cast<const int32_t*>(p)
            : *static_cast<const int64_t*>(p);
  }

  const std::vector<int64_t> pred_shape = ShapeOf(predictions.get());
  if (pred_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: InTopK expects rank-2 predictions.");
    return;
  }
  const int64_t batch = pred_shape[0];
  const int64_t classes = pred_shape[1];
  const std::vector<int64_t> out_shape = {batch};

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_BOOL, out_shape.data(), 1,
      static_cast<size_t>(batch) * TF_DataTypeSize(TF_BOOL), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (batch == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(TF_FLOAT, &mps_dtype, status)) return;
  MPSDataType idx_dtype;
  if (!MPSTypeFor(TF_TensorType(targets.get()), &idx_dtype, status)) return;

  std::string key = "InTopK";
  AppendShapeToKey(pred_shape, &key);
  key.append("/k").append(std::to_string(k));
  key.append("/i").append(
      std::to_string(static_cast<int>(TF_TensorType(targets.get()))));
  const std::vector<int64_t> target_shape = ShapeOf(targets.get());
  const NSUInteger depth = static_cast<NSUInteger>(classes);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* p = [g placeholderWithShape:MPSShape(pred_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* t = [g placeholderWithShape:MPSShape(target_shape)
                                           dataType:idx_dtype
                                               name:nil];
        // The target's own score, picked out with a one-hot mask.
        MPSGraphTensor* hot =
            [g oneHotWithIndicesTensor:[g castTensor:t
                                              toType:MPSDataTypeInt32
                                                name:nil]
                                 depth:depth
                                  axis:1
                              dataType:mps_dtype
                               onValue:1.0
                              offValue:0.0
                                  name:nil];
        MPSGraphTensor* target_score =
            [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:p
                                                        secondaryTensor:hot
                                                                   name:nil]
                                 axis:1
                                 name:nil];
        // Strictly greater, so ties do not count against the target, which is
        // what TensorFlow specifies.
        MPSGraphTensor* beats =
            [g greaterThanWithPrimaryTensor:p
                            secondaryTensor:target_score
                                       name:nil];
        MPSGraphTensor* n_beats =
            [g reductionSumWithTensor:[g castTensor:beats
                                             toType:MPSDataTypeInt32
                                               name:nil]
                                 axis:1
                                 name:nil];
        MPSGraphTensor* in_top =
            [g lessThanWithPrimaryTensor:n_beats
                         secondaryTensor:[g constantWithScalar:(double)k
                                                      dataType:MPSDataTypeInt32]
                                    name:nil];
        [out->inputs addObject:p];
        [out->inputs addObject:t];
        [out->outputs addObject:[g reshapeTensor:[g castTensor:in_top
                                                        toType:MPSDataTypeBool
                                                          name:nil]
                                       withShape:MPSShape(out_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* p_data =
      TensorDataForTensor(predictions.get(), TF_FLOAT, device, status);
  if (p_data == nil) return;
  MPSGraphTensorData* t_data = TensorDataForTensor(
      targets.get(), TF_TensorType(targets.get()), device, status);
  if (t_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_BOOL, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ p_data, t_data ], @[ o_data ], status);
}

void InTopK_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<TopKOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: InTopK kernel has no state.");
  } else {
    InTopK_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
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

METAL_COMPUTE(Equal_Compute, DTypeOp, Compare_ComputeImpl<CompareKind::kEqual>)
METAL_COMPUTE(NotEqual_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kNotEqual>)
METAL_COMPUTE(Less_Compute, DTypeOp, Compare_ComputeImpl<CompareKind::kLess>)
METAL_COMPUTE(LessEqual_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kLessEqual>)
METAL_COMPUTE(Greater_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kGreater>)
METAL_COMPUTE(GreaterEqual_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kGreaterEqual>)
METAL_COMPUTE(ApproxEqual_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kApproximateEqual>)
METAL_COMPUTE(LogicalAnd_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kLogicalAnd>)
METAL_COMPUTE(LogicalOr_Compute, DTypeOp,
              Compare_ComputeImpl<CompareKind::kLogicalOr>)
METAL_COMPUTE(LogicalNot_Compute, DTypeOp, LogicalNot_ComputeImpl)
METAL_COMPUTE(Select_Compute, DTypeOp, Select_ComputeImpl)
METAL_COMPUTE(ArgMax_Compute, ArgOp, Arg_ComputeImpl<true>)
METAL_COMPUTE(ArgMin_Compute, ArgOp, Arg_ComputeImpl<false>)

#undef METAL_COMPUTE

void RegisterInTopK(const char* op_name, TF_DataType index_dtype,
                    bool k_on_host, const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &TopKOp_Create, &InTopK_Compute,
      &TopKOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", index_dtype, status);
  if (k_on_host) TF_KernelBuilder_HostMemory(builder, "k");
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

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), void (*destroy)(void*),
              const std::string& name, TF_DataType dtype, bool constrain_t,
              const char* host_arg = nullptr, const char* attr2 = nullptr,
              TF_DataType dtype2 = TF_INT64) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
  if (constrain_t) TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalCompareKernels() {
  // InTopK's k is an attribute; InTopKV2 takes it as a host tensor. T here
  // constrains the target index type, not the predictions, which are float.
  {
    static constexpr TF_DataType kIdx[] = {TF_INT32, TF_INT64};
    static constexpr const char* kIdxName[] = {"Int32", "Int64"};
    for (int j = 0; j < 2; ++j) {
      RegisterInTopK("InTopK", kIdx[j], /*k_on_host=*/false,
                     std::string("MetalInTopK") + kIdxName[j]);
      RegisterInTopK("InTopKV2", kIdx[j], /*k_on_host=*/true,
                     std::string("MetalInTopKV2") + kIdxName[j]);
    }
  }

  struct Entry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
  };
  static const Entry kComparisons[] = {
      {"Equal", &Equal_Compute},           {"NotEqual", &NotEqual_Compute},
      {"Less", &Less_Compute},             {"LessEqual", &LessEqual_Compute},
      {"Greater", &Greater_Compute},       {"GreaterEqual", &GreaterEqual_Compute},
  };
  // Comparisons are useful over the index types too, not just floats: an
  // accuracy metric compares two int64 label vectors.
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF, TF_INT32,
                                            TF_INT64};
  static constexpr const char* kSuffixes[] = {"Float", "Half", "Int32",
                                              "Int64"};
  for (int i = 0; i < 4; ++i) {
    for (const Entry& e : kComparisons) {
      Register(e.op, &DTypeOp_Create, e.compute, &DTypeOp_Delete,
               std::string("Metal") + e.op + kSuffixes[i], kDTypes[i], true);
    }
    Register("Select", &DTypeOp_Create, &Select_Compute, &DTypeOp_Delete,
             std::string("MetalSelect") + kSuffixes[i], kDTypes[i], true);
    Register("SelectV2", &DTypeOp_Create, &Select_Compute, &DTypeOp_Delete,
             std::string("MetalSelectV2") + kSuffixes[i], kDTypes[i], true);
  }

  // ApproximateEqual's op definition allows only floating point, so it is
  // registered outside the loop above rather than over the index types too.
  for (int i = 0; i < 2; ++i) {
    Register("ApproximateEqual", &DTypeOp_Create, &ApproxEqual_Compute,
             &DTypeOp_Delete,
             std::string("MetalApproximateEqual") + kSuffixes[i], kDTypes[i],
             true);
  }

  // The logical operators take bool on both sides, so T is not constrained.
  Register("LogicalAnd", &DTypeOp_Create, &LogicalAnd_Compute, &DTypeOp_Delete,
           "MetalLogicalAnd", TF_BOOL, false);
  Register("LogicalOr", &DTypeOp_Create, &LogicalOr_Compute, &DTypeOp_Delete,
           "MetalLogicalOr", TF_BOOL, false);
  Register("LogicalNot", &DTypeOp_Create, &LogicalNot_Compute, &DTypeOp_Delete,
           "MetalLogicalNot", TF_BOOL, false);

  // The reduction dimension is read on the host to build the output shape.
  static constexpr TF_DataType kOutTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kOutSuffixes[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      Register("ArgMax", &ArgOp_Create, &ArgMax_Compute, &ArgOp_Delete,
               std::string("MetalArgMax") + kSuffixes[i] + kOutSuffixes[j],
               kDTypes[i], true, "dimension", "output_type", kOutTypes[j]);
      Register("ArgMin", &ArgOp_Create, &ArgMin_Compute, &ArgOp_Delete,
               std::string("MetalArgMin") + kSuffixes[i] + kOutSuffixes[j],
               kDTypes[i], true, "dimension", "output_type", kOutTypes[j]);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
