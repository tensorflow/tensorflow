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

// Elementwise arithmetic on MPSGraph.
//
// An earlier version of this file used hand-written compute shaders and could
// only broadcast a scalar operand. That is not a limitation real graphs
// tolerate: gradients broadcast against reduced axes constantly. MPSGraph
// applies NumPy broadcasting natively, so moving these ops onto it removes the
// restriction rather than working around it, and brings the rest of the maths
// library with it.

// NumPy broadcasting: right-align the shapes, and each pair of extents must be
// equal or one of them must be 1.
bool BroadcastShape(const std::vector<int64_t>& lhs,
                    const std::vector<int64_t>& rhs,
                    std::vector<int64_t>* out, TF_Status* status) {
  const size_t rank = std::max(lhs.size(), rhs.size());
  out->assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t a = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const int64_t b = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (a != b && a != 1 && b != 1) {
      auto describe = [](const std::vector<int64_t>& s) {
        std::string t = "[";
        for (size_t j = 0; j < s.size(); ++j) {
          if (j > 0) t += ",";
          t += std::to_string(s[j]);
        }
        return t + "]";
      };
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   ("Metal: shapes " + describe(lhs) + " and " + describe(rhs) +
                    " do not broadcast.")
                       .c_str());
      return false;
    }
    (*out)[i] = std::max(a, b);
  }
  return true;
}

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

/*** BINARY ***/

enum class BinaryKind {
  kAdd, kSub, kMul, kDiv, kMaximum, kMinimum, kPow, kSquaredDifference,
  kFloorDiv, kFloorMod, kMod, kAtan2, kXdivy, kXlogy,
};

const char* NameOf(BinaryKind k) {
  switch (k) {
    case BinaryKind::kAdd: return "Add";
    case BinaryKind::kSub: return "Sub";
    case BinaryKind::kMul: return "Mul";
    case BinaryKind::kDiv: return "Div";
    case BinaryKind::kMaximum: return "Maximum";
    case BinaryKind::kMinimum: return "Minimum";
    case BinaryKind::kPow: return "Pow";
    case BinaryKind::kSquaredDifference: return "SquaredDifference";
    case BinaryKind::kFloorDiv: return "FloorDiv";
    case BinaryKind::kFloorMod: return "FloorMod";
    case BinaryKind::kMod: return "Mod";
    case BinaryKind::kAtan2: return "Atan2";
    case BinaryKind::kXdivy: return "Xdivy";
    case BinaryKind::kXlogy: return "Xlogy";
  }
  return "?";
}

MPSGraphTensor* ApplyBinary(MPSGraph* g, BinaryKind k, MPSGraphTensor* a,
                            MPSGraphTensor* b, MPSDataType t) {
  switch (k) {
    case BinaryKind::kAdd:
      return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kSub:
      return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kMul:
      return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kDiv:
      return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kMaximum:
      return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kMinimum:
      return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kPow:
      return [g powerWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kSquaredDifference: {
      MPSGraphTensor* d =
          [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
      return [g squareWithTensor:d name:nil];
    }
    case BinaryKind::kFloorDiv: {
      MPSGraphTensor* d =
          [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
      return [g floorWithTensor:d name:nil];
    }
    case BinaryKind::kFloorMod:
      // Floored modulo takes the sign of the divisor, which is what
      // TensorFlow's FloorMod means and what Mod below does not.
      return [g floorModuloWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kMod:
      return [g moduloWithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kAtan2:
      return [g atan2WithPrimaryTensor:a secondaryTensor:b name:nil];
    case BinaryKind::kXdivy: {
      // x/y, but 0 wherever x is 0, even where y is 0 too.
      MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
      MPSGraphTensor* is_zero =
          [g equalWithPrimaryTensor:a secondaryTensor:zero name:nil];
      return [g selectWithPredicateTensor:is_zero
                      truePredicateTensor:zero
                     falsePredicateTensor:
                         [g divisionWithPrimaryTensor:a
                                      secondaryTensor:b
                                                 name:nil]
                                     name:nil];
    }
    case BinaryKind::kXlogy: {
      // x*log(y), but 0 wherever x is 0, even where log(y) is -inf.
      MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
      MPSGraphTensor* is_zero =
          [g equalWithPrimaryTensor:a secondaryTensor:zero name:nil];
      MPSGraphTensor* prod =
          [g multiplicationWithPrimaryTensor:a
                             secondaryTensor:[g logarithmWithTensor:b name:nil]
                                        name:nil];
      return [g selectWithPredicateTensor:is_zero
                      truePredicateTensor:zero
                     falsePredicateTensor:prod
                                     name:nil];
    }
  }
  return nil;
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

template <BinaryKind kKind>
void Binary_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor lhs;
  ScopedTensor rhs;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> lhs_shape = ShapeOf(lhs.get());
  const std::vector<int64_t> rhs_shape = ShapeOf(rhs.get());
  std::vector<int64_t> out_shape;
  if (!BroadcastShape(lhs_shape, rhs_shape, &out_shape, status)) return;

  ScopedTensor output;
  const int64_t count = ElementCount(out_shape);
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

  std::string key = NameOf(kKind);
  AppendShapeToKey(lhs_shape, &key);
  AppendShapeToKey(rhs_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* a = [out->graph placeholderWithShape:MPSShape(lhs_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* b = [out->graph placeholderWithShape:MPSShape(rhs_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs addObject:ApplyBinary(out->graph, kKind, a, b, mps_dtype)];
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
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ a_data, b_data ], @[ out_data ], status);
}

/*** UNARY ***/

enum class UnaryKind {
  kNeg, kSqrt, kRsqrt, kExp, kLog, kSquare, kTanh, kSigmoid, kAbs, kReciprocal,
  kFloor, kCeil, kRound, kRint, kSign, kErf, kSoftplus, kElu, kSelu, kLog1p,
  kExpm1,
  kSin, kCos, kTan, kAsin, kAcos, kAtan, kSinh, kCosh, kAsinh, kAcosh, kAtanh,
};

const char* NameOf(UnaryKind k) {
  switch (k) {
    case UnaryKind::kNeg: return "Neg";
    case UnaryKind::kSqrt: return "Sqrt";
    case UnaryKind::kRsqrt: return "Rsqrt";
    case UnaryKind::kExp: return "Exp";
    case UnaryKind::kLog: return "Log";
    case UnaryKind::kSquare: return "Square";
    case UnaryKind::kTanh: return "Tanh";
    case UnaryKind::kSigmoid: return "Sigmoid";
    case UnaryKind::kAbs: return "Abs";
    case UnaryKind::kReciprocal: return "Reciprocal";
    case UnaryKind::kFloor: return "Floor";
    case UnaryKind::kCeil: return "Ceil";
    case UnaryKind::kRound: return "Round";
    case UnaryKind::kRint: return "Rint";
    case UnaryKind::kSign: return "Sign";
    case UnaryKind::kErf: return "Erf";
    case UnaryKind::kSoftplus: return "Softplus";
    case UnaryKind::kElu: return "Elu";
    case UnaryKind::kSelu: return "Selu";
    case UnaryKind::kLog1p: return "Log1p";
    case UnaryKind::kExpm1: return "Expm1";
    case UnaryKind::kSin: return "Sin";
    case UnaryKind::kCos: return "Cos";
    case UnaryKind::kTan: return "Tan";
    case UnaryKind::kAsin: return "Asin";
    case UnaryKind::kAcos: return "Acos";
    case UnaryKind::kAtan: return "Atan";
    case UnaryKind::kSinh: return "Sinh";
    case UnaryKind::kCosh: return "Cosh";
    case UnaryKind::kAsinh: return "Asinh";
    case UnaryKind::kAcosh: return "Acosh";
    case UnaryKind::kAtanh: return "Atanh";
  }
  return "?";
}

// Softplus, written the stable way TensorFlow writes it:
//   max(x, 0) + log(1 + exp(-|x|))
// Rather than log(1 + exp(x)), which overflows for large positive x.
MPSGraphTensor* Softplus(MPSGraph* g, MPSGraphTensor* x, MPSDataType t) {
  MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
  MPSGraphTensor* pos = [g maximumWithPrimaryTensor:x secondaryTensor:zero name:nil];
  MPSGraphTensor* negabs =
      [g negativeWithTensor:[g absoluteWithTensor:x name:nil] name:nil];
  MPSGraphTensor* e = [g exponentWithTensor:negabs name:nil];
  MPSGraphTensor* l =
      [g logarithmWithTensor:[g additionWithPrimaryTensor:one
                                         secondaryTensor:e
                                                    name:nil]
                        name:nil];
  return [g additionWithPrimaryTensor:pos secondaryTensor:l name:nil];
}

// Elu: x for x > 0, exp(x) - 1 otherwise. Built from a select rather than a
// clamp so the negative branch keeps its exact exponential shape.
MPSGraphTensor* Elu(MPSGraph* g, MPSGraphTensor* x, MPSDataType t) {
  MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
  MPSGraphTensor* neg =
      [g subtractionWithPrimaryTensor:[g exponentWithTensor:x name:nil]
                      secondaryTensor:one
                                 name:nil];
  MPSGraphTensor* mask =
      [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];
  return [g selectWithPredicateTensor:mask
                  truePredicateTensor:x
                 falsePredicateTensor:neg
                                 name:nil];
}

MPSGraphTensor* ApplyUnary(MPSGraph* g, UnaryKind k, MPSGraphTensor* x,
                           MPSDataType t) {
  switch (k) {
    case UnaryKind::kNeg: return [g negativeWithTensor:x name:nil];
    case UnaryKind::kSqrt: return [g squareRootWithTensor:x name:nil];
    case UnaryKind::kRsqrt: return [g reciprocalSquareRootWithTensor:x name:nil];
    case UnaryKind::kExp: return [g exponentWithTensor:x name:nil];
    case UnaryKind::kLog: return [g logarithmWithTensor:x name:nil];
    case UnaryKind::kSquare: return [g squareWithTensor:x name:nil];
    case UnaryKind::kTanh: return [g tanhWithTensor:x name:nil];
    case UnaryKind::kSigmoid: return [g sigmoidWithTensor:x name:nil];
    case UnaryKind::kAbs: return [g absoluteWithTensor:x name:nil];
    case UnaryKind::kReciprocal: return [g reciprocalWithTensor:x name:nil];
    case UnaryKind::kFloor: return [g floorWithTensor:x name:nil];
    case UnaryKind::kCeil: return [g ceilWithTensor:x name:nil];
    // TensorFlow's Round rounds halves to even, which is rint, not round:
    // MPSGraph's roundWithTensor rounds halves away from zero and would send
    // 2.5 to 3 where TensorFlow sends it to 2.
    case UnaryKind::kRound: return [g rintWithTensor:x name:nil];
    case UnaryKind::kRint: return [g rintWithTensor:x name:nil];
    case UnaryKind::kSign: return [g signWithTensor:x name:nil];
    case UnaryKind::kErf: return [g erfWithTensor:x name:nil];
    case UnaryKind::kSoftplus: return Softplus(g, x, t);
    case UnaryKind::kElu: return Elu(g, x, t);
    case UnaryKind::kSelu: {
      // The fixed constants from the self-normalising networks paper, which
      // is what TensorFlow's Selu uses.
      MPSGraphTensor* alpha =
          [g constantWithScalar:1.6732632423543772 dataType:t];
      MPSGraphTensor* scale =
          [g constantWithScalar:1.0507009873554805 dataType:t];
      MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
      MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
      MPSGraphTensor* neg = [g
          multiplicationWithPrimaryTensor:alpha
                          secondaryTensor:
                              [g subtractionWithPrimaryTensor:
                                     [g exponentWithTensor:x name:nil]
                                              secondaryTensor:one
                                                         name:nil]
                                     name:nil];
      MPSGraphTensor* mask =
          [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];
      MPSGraphTensor* branch = [g selectWithPredicateTensor:mask
                                        truePredicateTensor:x
                                       falsePredicateTensor:neg
                                                       name:nil];
      return [g multiplicationWithPrimaryTensor:scale
                                secondaryTensor:branch
                                           name:nil];
    }
    case UnaryKind::kLog1p: {
      MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
      return [g logarithmWithTensor:[g additionWithPrimaryTensor:x
                                                secondaryTensor:one
                                                           name:nil]
                               name:nil];
    }
    case UnaryKind::kExpm1: {
      MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
      return [g subtractionWithPrimaryTensor:[g exponentWithTensor:x name:nil]
                             secondaryTensor:one
                                        name:nil];
    }
    case UnaryKind::kSin: return [g sinWithTensor:x name:nil];
    case UnaryKind::kCos: return [g cosWithTensor:x name:nil];
    case UnaryKind::kTan: return [g tanWithTensor:x name:nil];
    case UnaryKind::kAsin: return [g asinWithTensor:x name:nil];
    case UnaryKind::kAcos: return [g acosWithTensor:x name:nil];
    case UnaryKind::kAtan: return [g atanWithTensor:x name:nil];
    case UnaryKind::kSinh: return [g sinhWithTensor:x name:nil];
    case UnaryKind::kCosh: return [g coshWithTensor:x name:nil];
    case UnaryKind::kAsinh: return [g asinhWithTensor:x name:nil];
    case UnaryKind::kAcosh: return [g acoshWithTensor:x name:nil];
    case UnaryKind::kAtanh: return [g atanhWithTensor:x name:nil];
  }
  return nil;
}

template <UnaryKind kKind>
void Unary_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
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

  std::string key = NameOf(kKind);
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:ApplyUnary(out->graph, kKind, x, mps_dtype)];
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

/*** UNARY GRADIENTS ***/

// TensorFlow's *Grad ops take the forward output y and the incoming gradient
// dy, not the forward input, so each formula is expressed in terms of y.
enum class UnaryGradKind { kTanh, kSigmoid, kSqrt, kRsqrt };

const char* NameOf(UnaryGradKind k) {
  switch (k) {
    case UnaryGradKind::kTanh: return "TanhGrad";
    case UnaryGradKind::kSigmoid: return "SigmoidGrad";
    case UnaryGradKind::kSqrt: return "SqrtGrad";
    case UnaryGradKind::kRsqrt: return "RsqrtGrad";
  }
  return "?";
}

MPSGraphTensor* ApplyUnaryGrad(MPSGraph* g, UnaryGradKind k, MPSGraphTensor* y,
                               MPSGraphTensor* dy, MPSDataType dtype) {
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dtype];
  switch (k) {
    case UnaryGradKind::kTanh: {
      // dy * (1 - y^2)
      MPSGraphTensor* y2 = [g squareWithTensor:y name:nil];
      MPSGraphTensor* t =
          [g subtractionWithPrimaryTensor:one secondaryTensor:y2 name:nil];
      return [g multiplicationWithPrimaryTensor:dy secondaryTensor:t name:nil];
    }
    case UnaryGradKind::kSigmoid: {
      // dy * y * (1 - y)
      MPSGraphTensor* t =
          [g subtractionWithPrimaryTensor:one secondaryTensor:y name:nil];
      MPSGraphTensor* u =
          [g multiplicationWithPrimaryTensor:y secondaryTensor:t name:nil];
      return [g multiplicationWithPrimaryTensor:dy secondaryTensor:u name:nil];
    }
    case UnaryGradKind::kSqrt: {
      // dy * 0.5 / y
      MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dtype];
      MPSGraphTensor* t =
          [g multiplicationWithPrimaryTensor:dy secondaryTensor:half name:nil];
      return [g divisionWithPrimaryTensor:t secondaryTensor:y name:nil];
    }
    case UnaryGradKind::kRsqrt: {
      // dy * -0.5 * y^3
      MPSGraphTensor* mhalf = [g constantWithScalar:-0.5 dataType:dtype];
      MPSGraphTensor* y2 = [g squareWithTensor:y name:nil];
      MPSGraphTensor* y3 =
          [g multiplicationWithPrimaryTensor:y2 secondaryTensor:y name:nil];
      MPSGraphTensor* t =
          [g multiplicationWithPrimaryTensor:dy secondaryTensor:mhalf name:nil];
      return [g multiplicationWithPrimaryTensor:t secondaryTensor:y3 name:nil];
    }
  }
  return nil;
}

template <UnaryGradKind kKind>
void UnaryGrad_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor y;
  ScopedTensor dy;
  TF_GetInput(ctx, 0, y.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, dy.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(y.get());
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

  std::string key = NameOf(kKind);
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* a = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* b = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs
            addObject:ApplyUnaryGrad(out->graph, kKind, a, b, mps_dtype)];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* y_data =
      TensorDataForTensor(y.get(), op->dtype, device, status);
  if (y_data == nil) return;
  MPSGraphTensorData* dy_data =
      TensorDataForTensor(dy.get(), op->dtype, device, status);
  if (dy_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ y_data, dy_data ], @[ out_data ], status);
}

/*** CAST ***/

struct CastOp {
  TF_DataType src = TF_FLOAT;
  TF_DataType dst = TF_FLOAT;
};

void* CastOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new CastOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "SrcT", &op->src, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "DstT", &op->dst, status);
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

void CastOp_Delete(void* kernel) { delete static_cast<CastOp*>(kernel); }

void Cast_ComputeImpl(CastOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dst, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dst), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType src_type;
  MPSDataType dst_type;
  if (!MPSTypeFor(op->src, &src_type, status)) return;
  if (!MPSTypeFor(op->dst, &dst_type, status)) return;

  std::string key = "Cast";
  AppendShapeToKey(shape, &key);
  key.append("/").append(std::to_string(static_cast<int>(op->src)));
  key.append("->").append(std::to_string(static_cast<int>(op->dst)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:src_type
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph castTensor:x
                                                toType:dst_type
                                                  name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->src, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dst, device, status);
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

METAL_COMPUTE(Add_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kAdd>)
METAL_COMPUTE(Sub_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kSub>)
METAL_COMPUTE(Mul_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kMul>)
METAL_COMPUTE(Div_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kDiv>)
METAL_COMPUTE(Max_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kMaximum>)
METAL_COMPUTE(Min_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kMinimum>)
METAL_COMPUTE(Pow_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kPow>)
METAL_COMPUTE(SqDiff_Compute, DTypeOp,
              Binary_ComputeImpl<BinaryKind::kSquaredDifference>)
METAL_COMPUTE(FloorDiv_Compute, DTypeOp,
              Binary_ComputeImpl<BinaryKind::kFloorDiv>)
METAL_COMPUTE(FloorMod_Compute, DTypeOp,
              Binary_ComputeImpl<BinaryKind::kFloorMod>)
METAL_COMPUTE(Mod_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kMod>)

METAL_COMPUTE(Neg_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kNeg>)
METAL_COMPUTE(Sqrt_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSqrt>)
METAL_COMPUTE(Rsqrt_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kRsqrt>)
METAL_COMPUTE(Exp_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kExp>)
METAL_COMPUTE(Log_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kLog>)
METAL_COMPUTE(Square_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSquare>)
METAL_COMPUTE(Tanh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kTanh>)
METAL_COMPUTE(Sigmoid_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSigmoid>)
METAL_COMPUTE(Abs_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAbs>)
METAL_COMPUTE(Recip_Compute, DTypeOp,
              Unary_ComputeImpl<UnaryKind::kReciprocal>)
METAL_COMPUTE(Floor_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kFloor>)
METAL_COMPUTE(Ceil_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kCeil>)
METAL_COMPUTE(Round_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kRound>)
METAL_COMPUTE(Rint_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kRint>)
METAL_COMPUTE(Sign_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSign>)
METAL_COMPUTE(Erf_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kErf>)
METAL_COMPUTE(Expm1_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kExpm1>)
METAL_COMPUTE(Log1p_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kLog1p>)
METAL_COMPUTE(Sin_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSin>)
METAL_COMPUTE(Cos_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kCos>)
METAL_COMPUTE(Tan_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kTan>)
METAL_COMPUTE(Asin_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAsin>)
METAL_COMPUTE(Acos_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAcos>)
METAL_COMPUTE(Atan_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAtan>)
METAL_COMPUTE(Sinh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSinh>)
METAL_COMPUTE(Cosh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kCosh>)
METAL_COMPUTE(Asinh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAsinh>)
METAL_COMPUTE(Acosh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAcosh>)
METAL_COMPUTE(Atanh_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kAtanh>)
METAL_COMPUTE(Softplus_Compute, DTypeOp,
              Unary_ComputeImpl<UnaryKind::kSoftplus>)
METAL_COMPUTE(Elu_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kElu>)
METAL_COMPUTE(Selu_Compute, DTypeOp, Unary_ComputeImpl<UnaryKind::kSelu>)
METAL_COMPUTE(TanhGrad_Compute, DTypeOp,
              UnaryGrad_ComputeImpl<UnaryGradKind::kTanh>)
METAL_COMPUTE(SigmoidGrad_Compute, DTypeOp,
              UnaryGrad_ComputeImpl<UnaryGradKind::kSigmoid>)
METAL_COMPUTE(SqrtGrad_Compute, DTypeOp,
              UnaryGrad_ComputeImpl<UnaryGradKind::kSqrt>)
METAL_COMPUTE(RsqrtGrad_Compute, DTypeOp,
              UnaryGrad_ComputeImpl<UnaryGradKind::kRsqrt>)
METAL_COMPUTE(Atan2_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kAtan2>)
METAL_COMPUTE(Xdivy_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kXdivy>)
METAL_COMPUTE(Xlogy_Compute, DTypeOp, Binary_ComputeImpl<BinaryKind::kXlogy>)

METAL_COMPUTE(Cast_Compute, CastOp, Cast_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), void (*destroy)(void*),
              const char* attr, TF_DataType dtype, const std::string& name,
              const char* attr2 = nullptr, TF_DataType dtype2 = TF_FLOAT) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
  TF_KernelBuilder_TypeConstraint(builder, attr, dtype, status);
  if (TF_GetCode(status) == TF_OK && attr2 != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, attr2, dtype2, status);
  }
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

void RegisterMetalElementwiseKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};

  struct BinaryEntry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
  };
  // AddV2 is what modern graphs emit; Add is kept for graphs still carrying
  // the v1 op. RealDiv and Div are the same operation on floating point.
  static const BinaryEntry kBinaries[] = {
      {"AddV2", &Add_Compute},   {"Add", &Add_Compute},
      {"Sub", &Sub_Compute},     {"Mul", &Mul_Compute},
      {"Div", &Div_Compute},     {"RealDiv", &Div_Compute},
      {"FloorDiv", &FloorDiv_Compute}, {"FloorMod", &FloorMod_Compute},
      {"Mod", &Mod_Compute},
      {"Maximum", &Max_Compute}, {"Minimum", &Min_Compute},
      {"Pow", &Pow_Compute},     {"SquaredDifference", &SqDiff_Compute},
      {"Atan2", &Atan2_Compute},
      {"Xdivy", &Xdivy_Compute},       {"Xlogy", &Xlogy_Compute},
  };

  struct UnaryEntry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
  };
  static const UnaryEntry kUnaries[] = {
      {"Neg", &Neg_Compute},         {"Sqrt", &Sqrt_Compute},
      {"Rsqrt", &Rsqrt_Compute},     {"Exp", &Exp_Compute},
      {"Log", &Log_Compute},         {"Square", &Square_Compute},
      {"Tanh", &Tanh_Compute},       {"Sigmoid", &Sigmoid_Compute},
      {"Abs", &Abs_Compute},         {"Reciprocal", &Recip_Compute},
      {"Floor", &Floor_Compute},     {"Ceil", &Ceil_Compute},
      {"Round", &Round_Compute},     {"Rint", &Rint_Compute},
      {"Sign", &Sign_Compute},       {"Erf", &Erf_Compute},
      {"Softplus", &Softplus_Compute}, {"Elu", &Elu_Compute},
      {"Selu", &Selu_Compute},       {"Log1p", &Log1p_Compute},
      {"Expm1", &Expm1_Compute},
      {"TanhGrad", &TanhGrad_Compute},
      {"SigmoidGrad", &SigmoidGrad_Compute},
      {"SqrtGrad", &SqrtGrad_Compute},
      {"RsqrtGrad", &RsqrtGrad_Compute},
      {"Sin", &Sin_Compute},       {"Cos", &Cos_Compute},
      {"Tan", &Tan_Compute},       {"Asin", &Asin_Compute},
      {"Acos", &Acos_Compute},     {"Atan", &Atan_Compute},
      {"Sinh", &Sinh_Compute},     {"Cosh", &Cosh_Compute},
      {"Asinh", &Asinh_Compute},   {"Acosh", &Acosh_Compute},
      {"Atanh", &Atanh_Compute},
  };

  for (int i = 0; i < 2; ++i) {
    const TF_DataType dtype = kDTypes[i];
    const std::string suffix = kSuffixes[i];
    for (const BinaryEntry& e : kBinaries) {
      Register(e.op, &DTypeOp_Create, e.compute, &DTypeOp_Delete, "T", dtype,
               std::string("Metal") + e.op + suffix);
    }
    for (const UnaryEntry& e : kUnaries) {
      Register(e.op, &DTypeOp_Create, e.compute, &DTypeOp_Delete, "T", dtype,
               std::string("Metal") + e.op + suffix);
    }
  }

  // Cast, over every pair of the types this backend represents.
  //
  // The list used to hold the handful a mixed-precision graph emits, which
  // left ordinary ones out: an optimiser casts its step counter from int64 to
  // float on every update, and with no kernel for that pair the value makes a
  // round trip through the host in the middle of the update.
  struct CastType { TF_DataType type; const char* name; };
  static const CastType kCastTypes[] = {
      {TF_FLOAT, "Float"}, {TF_HALF, "Half"},   {TF_BFLOAT16, "Bf"},
      {TF_INT32, "Int32"}, {TF_INT64, "Int64"}, {TF_BOOL, "Bool"},
      {TF_UINT8, "Uint8"}, {TF_INT8, "Int8"},
  };
  for (const CastType& from : kCastTypes) {
    for (const CastType& to : kCastTypes) {
      Register("Cast", &CastOp_Create, &Cast_Compute, &CastOp_Delete, "SrcT",
               from.type,
               std::string("MetalCast") + from.name + "To" + to.name, "DstT",
               to.type);
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
