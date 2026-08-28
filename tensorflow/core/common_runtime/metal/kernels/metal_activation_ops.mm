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

// LeakyRelu, the activation gradients that take an alpha or a scale, and
// BatchMatMul.
//
// Each gradient here is expressed in the variable TensorFlow actually hands
// it, which is not the same for every op: EluGrad and SeluGrad receive the
// forward output, while SoftplusGrad and LeakyReluGrad receive the forward
// input. Using the wrong one produces a gradient that is right near zero and
// wrong everywhere else.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct AlphaOp {
  TF_DataType dtype = TF_FLOAT;
  float alpha = 0.2f;  // TensorFlow's LeakyRelu default
};

void* AlphaOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new AlphaOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_OpKernelConstruction_GetAttrFloat(ctx, "alpha", &op->alpha, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->alpha = 0.2f;
  }
  TF_DeleteStatus(status);
  return op;
}

void AlphaOp_Delete(void* kernel) { delete static_cast<AlphaOp*>(kernel); }

enum class ActKind { kLeakyRelu, kLeakyReluGrad, kEluGrad, kSeluGrad,
                     kSoftplusGrad, kRelu6, kRelu6Grad, kSoftsign,
                     kSoftsignGrad, kLogSoftmax };

const char* NameOf(ActKind k) {
  switch (k) {
    case ActKind::kLeakyRelu: return "LeakyRelu";
    case ActKind::kLeakyReluGrad: return "LeakyReluGrad";
    case ActKind::kEluGrad: return "EluGrad";
    case ActKind::kSeluGrad: return "SeluGrad";
    case ActKind::kSoftplusGrad: return "SoftplusGrad";
    case ActKind::kRelu6: return "Relu6";
    case ActKind::kRelu6Grad: return "Relu6Grad";
    case ActKind::kSoftsign: return "Softsign";
    case ActKind::kSoftsignGrad: return "SoftsignGrad";
    case ActKind::kLogSoftmax: return "LogSoftmax";
  }
  return "?";
}

bool IsBinary(ActKind k) {
  return k != ActKind::kLeakyRelu && k != ActKind::kRelu6 &&
         k != ActKind::kSoftsign && k != ActKind::kLogSoftmax;
}

MPSGraphTensor* ApplyAct(MPSGraph* g, ActKind k, MPSGraphTensor* a,
                         MPSGraphTensor* b, MPSDataType t, float alpha) {
  MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:t];
  MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:t];
  switch (k) {
    case ActKind::kLeakyRelu:
      return [g leakyReLUWithTensor:a alpha:alpha name:nil];
    case ActKind::kLeakyReluGrad: {
      // a = incoming gradient, b = forward input.
      MPSGraphTensor* mask =
          [g greaterThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
      MPSGraphTensor* slope = [g selectWithPredicateTensor:mask
                                      truePredicateTensor:one
                                     falsePredicateTensor:
                                         [g constantWithScalar:alpha
                                                      dataType:t]
                                                     name:nil];
      return [g multiplicationWithPrimaryTensor:a
                                secondaryTensor:slope
                                           name:nil];
    }
    case ActKind::kEluGrad: {
      // a = incoming gradient, b = forward OUTPUT. d/dx elu = 1 for x > 0 and
      // exp(x) = y + 1 otherwise, which is why the output suffices.
      MPSGraphTensor* mask =
          [g greaterThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
      MPSGraphTensor* neg =
          [g additionWithPrimaryTensor:b secondaryTensor:one name:nil];
      MPSGraphTensor* slope = [g selectWithPredicateTensor:mask
                                      truePredicateTensor:one
                                     falsePredicateTensor:neg
                                                     name:nil];
      return [g multiplicationWithPrimaryTensor:a
                                secondaryTensor:slope
                                           name:nil];
    }
    case ActKind::kSeluGrad: {
      // a = incoming gradient, b = forward output.
      MPSGraphTensor* scale =
          [g constantWithScalar:1.0507009873554805 dataType:t];
      MPSGraphTensor* alpha_scale =
          [g constantWithScalar:1.7580993408473766 dataType:t];
      MPSGraphTensor* mask =
          [g greaterThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
      MPSGraphTensor* neg =
          [g additionWithPrimaryTensor:b secondaryTensor:alpha_scale name:nil];
      MPSGraphTensor* slope = [g selectWithPredicateTensor:mask
                                      truePredicateTensor:scale
                                     falsePredicateTensor:neg
                                                     name:nil];
      return [g multiplicationWithPrimaryTensor:a
                                secondaryTensor:slope
                                           name:nil];
    }
    case ActKind::kSoftplusGrad: {
      // a = incoming gradient, b = forward INPUT. d/dx softplus = sigmoid(x).
      return [g multiplicationWithPrimaryTensor:a
                                secondaryTensor:[g sigmoidWithTensor:b
                                                                name:nil]
                                           name:nil];
    }
    case ActKind::kRelu6:
      return [g clampWithTensor:a
                 minValueTensor:zero
                 maxValueTensor:[g constantWithScalar:6.0 dataType:t]
                           name:nil];
    case ActKind::kRelu6Grad: {
      // a = incoming gradient, b = forward input. The gradient passes only
      // strictly inside the clamp; at either bound Relu6 is flat.
      MPSGraphTensor* six = [g constantWithScalar:6.0 dataType:t];
      MPSGraphTensor* lo =
          [g greaterThanWithPrimaryTensor:b secondaryTensor:zero name:nil];
      MPSGraphTensor* hi =
          [g lessThanWithPrimaryTensor:b secondaryTensor:six name:nil];
      MPSGraphTensor* inside =
          [g logicalANDWithPrimaryTensor:lo secondaryTensor:hi name:nil];
      return [g selectWithPredicateTensor:inside
                      truePredicateTensor:a
                     falsePredicateTensor:zero
                                     name:nil];
    }
    case ActKind::kSoftsign: {
      // x / (1 + |x|)
      MPSGraphTensor* den =
          [g additionWithPrimaryTensor:one
                       secondaryTensor:[g absoluteWithTensor:a name:nil]
                                  name:nil];
      return [g divisionWithPrimaryTensor:a secondaryTensor:den name:nil];
    }
    case ActKind::kSoftsignGrad: {
      // a = incoming gradient, b = forward input. d/dx = 1/(1+|x|)^2.
      MPSGraphTensor* den =
          [g additionWithPrimaryTensor:one
                       secondaryTensor:[g absoluteWithTensor:b name:nil]
                                  name:nil];
      MPSGraphTensor* den2 = [g squareWithTensor:den name:nil];
      return [g divisionWithPrimaryTensor:a secondaryTensor:den2 name:nil];
    }
    case ActKind::kLogSoftmax: {
      // Written out rather than log(softMax(x)): a probability that underflows
      // to zero would give -inf, which is the whole reason LogSoftmax exists
      // as its own op.
      MPSGraphTensor* mx = [g reductionMaximumWithTensor:a axis:-1 name:nil];
      MPSGraphTensor* sh =
          [g subtractionWithPrimaryTensor:a secondaryTensor:mx name:nil];
      MPSGraphTensor* sm =
          [g reductionSumWithTensor:[g exponentWithTensor:sh name:nil]
                               axis:-1
                               name:nil];
      return [g subtractionWithPrimaryTensor:sh
                             secondaryTensor:[g logarithmWithTensor:sm
                                                               name:nil]
                                        name:nil];
    }
  }
  return nil;
}

template <ActKind kKind>
void Act_ComputeImpl(AlphaOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor first, second;
  TF_GetInput(ctx, 0, first.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (IsBinary(kKind)) {
    TF_GetInput(ctx, 1, second.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> shape = ShapeOf(first.get());
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
  key.append("/a").append(std::to_string(op->alpha));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const float alpha = op->alpha;
  const bool binary = IsBinary(kKind);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* a = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:a];
        MPSGraphTensor* b = nil;
        if (binary) {
          b = [out->graph placeholderWithShape:MPSShape(shape)
                                      dataType:mps_dtype
                                          name:nil];
          [out->inputs addObject:b];
        }
        [out->outputs
            addObject:ApplyAct(out->graph, kKind, a, b, mps_dtype, alpha)];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* a_data =
      TensorDataForTensor(first.get(), op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;

  if (binary) {
    MPSGraphTensorData* b_data =
        TensorDataForTensor(second.get(), op->dtype, device, status);
    if (b_data == nil) return;
    RunGraph(stream, *cached, @[ a_data, b_data ], @[ out_data ], status);
  } else {
    RunGraph(stream, *cached, @[ a_data ], @[ out_data ], status);
  }
}

/*** BATCH MATMUL ***/

struct BatchMatMulOp {
  TF_DataType dtype = TF_FLOAT;
  bool adj_x = false;
  bool adj_y = false;
};

void* BatchMatMulOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BatchMatMulOp();
  // V3 names the element types Ta, Tb and Tout rather than T.
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    TF_OpKernelConstruction_GetAttrType(ctx, "Ta", &op->dtype, status);
  }
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_Bool adj_x = 0, adj_y = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "adj_x", &adj_x, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrBool(ctx, "adj_y", &adj_y, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->adj_x = adj_x != 0;
  op->adj_y = adj_y != 0;
  TF_DeleteStatus(status);
  return op;
}

void BatchMatMulOp_Delete(void* kernel) {
  delete static_cast<BatchMatMulOp*>(kernel);
}

void BatchMatMul_ComputeImpl(BatchMatMulOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor lhs, rhs;
  TF_GetInput(ctx, 0, lhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, rhs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> a_shape = ShapeOf(lhs.get());
  const std::vector<int64_t> b_shape = ShapeOf(rhs.get());
  if (a_shape.size() < 2 || b_shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BatchMatMul needs operands of rank at least 2.");
    return;
  }

  const int64_t m = op->adj_x ? a_shape[a_shape.size() - 1]
                              : a_shape[a_shape.size() - 2];
  const int64_t ka = op->adj_x ? a_shape[a_shape.size() - 2]
                               : a_shape[a_shape.size() - 1];
  const int64_t kb = op->adj_y ? b_shape[b_shape.size() - 1]
                               : b_shape[b_shape.size() - 2];
  const int64_t n = op->adj_y ? b_shape[b_shape.size() - 2]
                              : b_shape[b_shape.size() - 1];
  if (ka != kb) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 ("Metal: BatchMatMul inner dimensions do not match: " +
                  std::to_string(ka) + " against " + std::to_string(kb) + ".")
                     .c_str());
    return;
  }

  // Batch dimensions broadcast, which is what V2 added over V1.
  const size_t a_batch = a_shape.size() - 2, b_batch = b_shape.size() - 2;
  const size_t batch_rank = std::max(a_batch, b_batch);
  std::vector<int64_t> out_shape(batch_rank + 2, 1);
  for (size_t i = 0; i < batch_rank; ++i) {
    const int64_t x = i < batch_rank - a_batch ? 1
                                               : a_shape[i - (batch_rank - a_batch)];
    const int64_t y = i < batch_rank - b_batch ? 1
                                               : b_shape[i - (batch_rank - b_batch)];
    if (x != y && x != 1 && y != 1) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: BatchMatMul batch dimensions do not broadcast.");
      return;
    }
    out_shape[i] = std::max(x, y);
  }
  out_shape[batch_rank] = m;
  out_shape[batch_rank + 1] = n;

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

  std::string key = "BatchMatMul";
  AppendShapeToKey(a_shape, &key);
  AppendShapeToKey(b_shape, &key);
  key.append(op->adj_x ? "/ax" : "/-").append(op->adj_y ? "ay" : "-");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const bool adj_x = op->adj_x, adj_y = op->adj_y;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* a = [g placeholderWithShape:MPSShape(a_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(b_shape)
                                           dataType:mps_dtype
                                               name:nil];
        // adj_x and adj_y transpose the last two dimensions. On real data an
        // adjoint is a transpose, which is all TensorFlow means here.
        MPSGraphTensor* at =
            adj_x ? [g transposeTensor:a
                             dimension:static_cast<NSUInteger>(a_shape.size() - 2)
                         withDimension:static_cast<NSUInteger>(a_shape.size() - 1)
                                  name:nil]
                  : a;
        MPSGraphTensor* bt =
            adj_y ? [g transposeTensor:b
                             dimension:static_cast<NSUInteger>(b_shape.size() - 2)
                         withDimension:static_cast<NSUInteger>(b_shape.size() - 1)
                                  name:nil]
                  : b;
        [out->inputs addObject:a];
        [out->inputs addObject:b];
        [out->outputs addObject:[g matrixMultiplicationWithPrimaryTensor:at
                                                         secondaryTensor:bt
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
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ a_data, b_data ], @[ out_data ], status);
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

METAL_COMPUTE(LeakyRelu_Compute, AlphaOp, Act_ComputeImpl<ActKind::kLeakyRelu>)
METAL_COMPUTE(LeakyReluGrad_Compute, AlphaOp,
              Act_ComputeImpl<ActKind::kLeakyReluGrad>)
METAL_COMPUTE(EluGrad_Compute, AlphaOp, Act_ComputeImpl<ActKind::kEluGrad>)
METAL_COMPUTE(SeluGrad_Compute, AlphaOp, Act_ComputeImpl<ActKind::kSeluGrad>)
METAL_COMPUTE(SoftplusGrad_Compute, AlphaOp,
              Act_ComputeImpl<ActKind::kSoftplusGrad>)
METAL_COMPUTE(Relu6_Compute, AlphaOp, Act_ComputeImpl<ActKind::kRelu6>)
METAL_COMPUTE(Relu6Grad_Compute, AlphaOp, Act_ComputeImpl<ActKind::kRelu6Grad>)
METAL_COMPUTE(Softsign_Compute, AlphaOp, Act_ComputeImpl<ActKind::kSoftsign>)
METAL_COMPUTE(SoftsignGrad_Compute, AlphaOp,
              Act_ComputeImpl<ActKind::kSoftsignGrad>)
METAL_COMPUTE(LogSoftmax_Compute, AlphaOp,
              Act_ComputeImpl<ActKind::kLogSoftmax>)
METAL_COMPUTE(BatchMatMul_Compute, BatchMatMulOp, BatchMatMul_ComputeImpl)

#undef METAL_COMPUTE

// `type_attrs` are the names the op gives its element types. Almost every op
// has a single T; BatchMatMulV3 has Ta, Tb and Tout instead, and constraining
// T there registers a kernel that can never match, which TensorFlow reports
// as "OpKernel 'BatchMatMulV3' has constraint on attr 'T' not in NodeDef".
void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), void (*destroy)(void*),
              TF_DataType dtype, const std::string& name,
              std::vector<const char*> type_attrs = {"T"}) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
  for (const char* attr : type_attrs) {
    if (TF_GetCode(status) != TF_OK) break;
    TF_KernelBuilder_TypeConstraint(builder, attr, dtype, status);
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

void RegisterMetalActivationKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};

  struct Entry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
  };
  static const Entry kActs[] = {
      {"LeakyRelu", &LeakyRelu_Compute},
      {"LeakyReluGrad", &LeakyReluGrad_Compute},
      {"EluGrad", &EluGrad_Compute},
      {"SeluGrad", &SeluGrad_Compute},
      {"SoftplusGrad", &SoftplusGrad_Compute},
      {"Relu6", &Relu6_Compute},
      {"Relu6Grad", &Relu6Grad_Compute},
      {"Softsign", &Softsign_Compute},
      {"SoftsignGrad", &SoftsignGrad_Compute},
      {"LogSoftmax", &LogSoftmax_Compute},
  };

  for (int i = 0; i < 2; ++i) {
    for (const Entry& e : kActs) {
      Register(e.op, &AlphaOp_Create, e.compute, &AlphaOp_Delete, kDTypes[i],
               std::string("Metal") + e.op + kSuffixes[i]);
    }
    // V1 has no batch broadcasting, V2 and V3 do; the implementation handles
    // both, so all three share it.
    Register("BatchMatMul", &BatchMatMulOp_Create, &BatchMatMul_Compute,
             &BatchMatMulOp_Delete, kDTypes[i],
             std::string("MetalBatchMatMul") + kSuffixes[i]);
    Register("BatchMatMulV2", &BatchMatMulOp_Create, &BatchMatMul_Compute,
             &BatchMatMulOp_Delete, kDTypes[i],
             std::string("MetalBatchMatMulV2") + kSuffixes[i]);
    Register("BatchMatMulV3", &BatchMatMulOp_Create, &BatchMatMul_Compute,
             &BatchMatMulOp_Delete, kDTypes[i],
             std::string("MetalBatchMatMulV3") + kSuffixes[i],
             {"Ta", "Tb", "Tout"});
  }
}

}  // namespace metal
}  // namespace tensorflow
