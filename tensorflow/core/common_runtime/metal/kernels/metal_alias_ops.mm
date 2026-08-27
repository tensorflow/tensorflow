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
#include <cstring>
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

// Ops that are a rename or a thin wrapper over something already implemented:
// the deprecated BatchMatrix* aliases, BiasAddV1, Conj, ConjugateTranspose,
// PopulationCount, InTopKV2 and Bucketize.
//
// The BatchMatrix* names are TensorFlow's pre-1.0 spelling of the Matrix*
// ops. They still appear in old graphs, and since the semantics are identical
// they are registered against the same implementations rather than left to
// fall back to the host.

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

struct AliasOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType out_dtype = TF_INT32;
  std::vector<float> boundaries;
};

void* AliasOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new AliasOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_OpKernelConstruction_GetAttrType(ctx, "out_type", &op->out_dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->out_dtype = TF_INT32;
  }
  // Bucketize's boundaries are a float attribute list of unknown length.
  int32_t total = 0;
  int32_t unused = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx, "boundaries", &total, &unused,
                                      status);
  if (TF_GetCode(status) == TF_OK && total > 0) {
    op->boundaries.resize(total);
    TF_OpKernelConstruction_GetAttrFloatList(ctx, "boundaries",
                                             op->boundaries.data(), total,
                                             status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      op->boundaries.clear();
    }
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
  TF_DeleteStatus(status);
  return op;
}

void AliasOp_Delete(void* kernel) { delete static_cast<AliasOp*>(kernel); }

/*** BIAS ADD V1 ***/

// BiasAddV1 is BiasAdd without the data_format attribute: the bias always
// applies to the last dimension.
void BiasAddV1_ComputeImpl(AliasOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor value, bias;
  TF_GetInput(ctx, 0, value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, bias.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(value.get());
  if (shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BiasAddV1 expects a rank of at least 1.");
    return;
  }
  std::vector<int64_t> bias_shape(shape.size(), 1);
  bias_shape.back() = TF_TensorElementCount(bias.get());

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

  std::string key = "BiasAddV1";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(bias_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* v = [out->graph placeholderWithShape:MPSShape(shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* b =
            [out->graph placeholderWithShape:MPSShape(bias_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:v];
        [out->inputs addObject:b];
        [out->outputs addObject:[out->graph additionWithPrimaryTensor:v
                                                     secondaryTensor:b
                                                                name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice bias_slice;
  if (!SliceForTensor(bias.get(), &bias_slice, status)) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(value.get(), op->dtype, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataFor(bias_slice, bias_shape, op->dtype, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ v_data, b_data ], @[ o_data ], status);
}

/*** CONJUGATE TRANSPOSE ***/

// On real tensors the conjugate is the identity, so this is Transpose. Complex
// dtypes are not supported by this backend at all, so there is no case where
// the distinction would matter here.
void ConjugateTranspose_ComputeImpl(AliasOp* op, TF_OpKernelContext* ctx,
                                    TF_Status* status) {
  ScopedTensor input, perm_t;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, perm_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int rank = static_cast<int>(in_shape.size());
  std::vector<int64_t> perm;
  if (!ReadHostVector(perm_t.get(), &perm, status)) return;
  if (static_cast<int>(perm.size()) != rank) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ConjugateTranspose permutation length does not match "
                 "the rank.");
    return;
  }
  std::vector<bool> seen(rank, false);
  std::vector<int64_t> out_shape(rank);
  for (int i = 0; i < rank; ++i) {
    int64_t a = perm[i];
    if (a < 0) a += rank;
    if (a < 0 || a >= rank || seen[a]) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: ConjugateTranspose permutation is not a "
                   "permutation.");
      return;
    }
    seen[a] = true;
    perm[i] = a;
    out_shape[i] = in_shape[a];
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

  std::string key = "ConjugateTranspose";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(perm, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  NSMutableArray<NSNumber*>* p = [NSMutableArray array];
  for (int64_t a : perm) [p addObject:@(static_cast<NSInteger>(a))];

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[out->graph transposeTensor:x
                                                permutation:p
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

/*** BUCKETIZE ***/

// Bucketize maps each value to the index of the interval it lands in. The
// boundaries are a compile-time attribute, so the whole thing is a sum of
// comparisons against constants.
void Bucketize_ComputeImpl(AliasOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_INT32, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_INT32), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Bucketize";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  for (float b : op->boundaries) key.append(",").append(std::to_string(b));
  const std::vector<float> boundaries = op->boundaries;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        // The bucket index is the count of boundaries the value is at or past,
        // which is TensorFlow's definition (upper bound, right-open).
        MPSGraphTensor* total =
            [g constantWithScalar:0.0 dataType:MPSDataTypeInt32];
        for (float b : boundaries) {
          MPSGraphTensor* ge =
              [g greaterThanOrEqualToWithPrimaryTensor:x
                                       secondaryTensor:
                                           [g constantWithScalar:b
                                                        dataType:mps_dtype]
                                                  name:nil];
          total = [g additionWithPrimaryTensor:total
                               secondaryTensor:[g castTensor:ge
                                                      toType:MPSDataTypeInt32
                                                        name:nil]
                                          name:nil];
        }
        [out->inputs addObject:x];
        [out->outputs addObject:total];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_INT32, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

/*** CONJ, POPULATION COUNT, CROSS ***/

// Conj is the identity on real data, for the same reason ConjugateTranspose
// is a plain transpose: this backend has no complex dtype.
void Conj_ComputeImpl(AliasOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_SetOutput(ctx, 0, input.get(), status);
}

// Cross product over the last axis, which must have length 3.
void Cross_ComputeImpl(AliasOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor a, b;
  TF_GetInput(ctx, 0, a.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, b.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(a.get());
  if (shape.empty() || shape.back() != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Cross needs a trailing axis of size 3.");
    return;
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

  std::string key = "Cross";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSUInteger axis = static_cast<NSUInteger>(shape.size()) - 1;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* y = [g placeholderWithShape:MPSShape(shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* comp[3][2];
        for (int i = 0; i < 3; ++i) {
          comp[i][0] = [g sliceTensor:x dimension:axis start:i length:1 name:nil];
          comp[i][1] = [g sliceTensor:y dimension:axis start:i length:1 name:nil];
        }
        MPSGraphTensor* parts[3];
        for (int i = 0; i < 3; ++i) {
          const int j = (i + 1) % 3, k = (i + 2) % 3;
          parts[i] = [g
              subtractionWithPrimaryTensor:
                  [g multiplicationWithPrimaryTensor:comp[j][0]
                                     secondaryTensor:comp[k][1]
                                                name:nil]
                           secondaryTensor:
                               [g multiplicationWithPrimaryTensor:comp[k][0]
                                                  secondaryTensor:comp[j][1]
                                                             name:nil]
                                      name:nil];
        }
        [out->inputs addObject:x];
        [out->inputs addObject:y];
        [out->outputs addObject:[g concatTensors:@[ parts[0], parts[1],
                                                    parts[2] ]
                                       dimension:static_cast<NSInteger>(axis)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* a_data =
      TensorDataForTensor(a.get(), op->dtype, device, status);
  if (a_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataForTensor(b.get(), op->dtype, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ a_data, b_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<AliasOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(BiasAddV1_Compute, BiasAddV1_ComputeImpl)
METAL_COMPUTE(ConjugateTranspose_Compute, ConjugateTranspose_ComputeImpl)
METAL_COMPUTE(Bucketize_Compute, Bucketize_ComputeImpl)
METAL_COMPUTE(Conj_Compute, Conj_ComputeImpl)
METAL_COMPUTE(Cross_Compute, Cross_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, std::vector<const char*> host_args) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &AliasOp_Create, compute, &AliasOp_Delete);
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

void RegisterMetalAliasKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    Register("BiasAddV1", &BiasAddV1_Compute, t, "MetalBiasAddV1" + s, {});
    Register("Bucketize", &Bucketize_Compute, t, "MetalBucketize" + s, {});
    Register("Conj", &Conj_Compute, t, "MetalConj" + s, {});
    Register("Cross", &Cross_Compute, t, "MetalCross" + s, {});
    for (int j = 0; j < 2; ++j) {
      Register("ConjugateTranspose", &ConjugateTranspose_Compute, t,
               "MetalConjugateTranspose" + s + kIndexSuffixes[j], {"perm"});
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
