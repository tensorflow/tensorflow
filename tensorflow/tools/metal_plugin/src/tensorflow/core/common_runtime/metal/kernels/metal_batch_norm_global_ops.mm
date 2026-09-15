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

// BatchNormWithGlobalNormalization and its gradient.
//
// This is the original, pre-fused batch normalisation: the statistics arrive
// as inputs rather than being computed, so there is nothing to reduce in the
// forward pass and the whole thing is elementwise with a broadcast over the
// channel axis. MPSGraph covers it directly.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct BatchNormOp {
  float epsilon = 0.0f;
  bool scale_after = true;
};

void* BatchNormOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BatchNormOp();
  float epsilon = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "variance_epsilon", &epsilon,
                                       status);
  if (TF_GetCode(status) == TF_OK) op->epsilon = epsilon;
  TF_SetStatus(status, TF_OK, "");
  TF_Bool scale = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "scale_after_normalization", &scale,
                                      status);
  if (TF_GetCode(status) == TF_OK) op->scale_after = scale != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void BatchNormOp_Delete(void* kernel) {
  delete static_cast<BatchNormOp*>(kernel);
}

// The per-channel vectors are rank 1 and the input is rank 4, so they are
// reshaped to broadcast along the channel axis.
MPSGraphTensor* OverChannels(MPSGraph* g, MPSGraphTensor* v, int64_t depth) {
  return [g reshapeTensor:v
                withShape:@[ @1, @1, @1, @(static_cast<NSInteger>(depth)) ]
                     name:nil];
}

void BatchNorm_ComputeImpl(BatchNormOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input, mean, variance, beta, gamma;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, mean.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, variance.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, beta.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, gamma.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BatchNormWithGlobalNormalization expects a rank-4 "
                 "input.");
    return;
  }
  const int64_t depth = shape[3];
  const std::vector<int64_t> vec_shape = {depth};
  const int64_t count = ElementCount(shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, shape.data(), 4,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "BatchNormGlobal";
  AppendShapeToKey(shape, &key);
  key.append("/e").append(std::to_string(op->epsilon));
  key.append(op->scale_after ? "/scaled" : "/plain");

  const float epsilon = op->epsilon;
  const bool scale_after = op->scale_after;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* m = [g placeholderWithShape:MPSShape(vec_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* v = [g placeholderWithShape:MPSShape(vec_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* b = [g placeholderWithShape:MPSShape(vec_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* gm = [g placeholderWithShape:MPSShape(vec_shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* eps =
            [g constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
        MPSGraphTensor* inv = [g
            reverseSquareRootWithTensor:[g additionWithPrimaryTensor:v
                                                    secondaryTensor:eps
                                                               name:nil]
                                   name:nil];
        if (scale_after) {
          inv = [g multiplicationWithPrimaryTensor:inv
                                   secondaryTensor:gm
                                              name:nil];
        }
        MPSGraphTensor* centred =
            [g subtractionWithPrimaryTensor:x
                            secondaryTensor:OverChannels(g, m, depth)
                                       name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:m];
        [out->inputs addObject:v];
        [out->inputs addObject:b];
        [out->inputs addObject:gm];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:
                             [g multiplicationWithPrimaryTensor:centred
                                                secondaryTensor:
                                                    OverChannels(g, inv, depth)
                                                           name:nil]
                                   secondaryTensor:OverChannels(g, b, depth)
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* m_data =
      TensorDataForTensor(mean.get(), TF_FLOAT, device, status);
  if (m_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(variance.get(), TF_FLOAT, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* b_data =
      TensorDataForTensor(beta.get(), TF_FLOAT, device, status);
  if (b_data == nil) return;
  MPSGraphTensorData* g_data =
      TensorDataForTensor(gamma.get(), TF_FLOAT, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, m_data, v_data, b_data, g_data ],
           @[ o_data ], status);
}

// Five outputs: the input gradient and one per statistic. The two reductions
// the formulas share are computed once, exactly as the CPU kernel shares its
// two scratch vectors.
void BatchNormGrad_ComputeImpl(BatchNormOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor input, mean, variance, gamma, backprop;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, mean.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, variance.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, gamma.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, backprop.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the batch normalisation gradient expects a rank-4 "
                 "input.");
    return;
  }
  const int64_t depth = shape[3];
  const std::vector<int64_t> vec_shape = {depth};
  const int64_t count = ElementCount(shape);
  const size_t vec_bytes = static_cast<size_t>(depth) * sizeof(float);

  ScopedTensor dx, dm, dv, db, dg;
  dx.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, shape.data(), 4,
                             static_cast<size_t>(count) * sizeof(float),
                             status));
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor* vectors[4] = {&dm, &dv, &db, &dg};
  for (int i = 0; i < 4; ++i) {
    vectors[i]->reset(TF_AllocateOutput(ctx, i + 1, TF_FLOAT, vec_shape.data(),
                                        1, vec_bytes, status));
    if (TF_GetCode(status) != TF_OK) return;
  }
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "BatchNormGlobalGrad";
  AppendShapeToKey(shape, &key);
  key.append("/e").append(std::to_string(op->epsilon));
  key.append(op->scale_after ? "/scaled" : "/plain");

  const float epsilon = op->epsilon;
  const bool scale_after = op->scale_after;
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* m = [g placeholderWithShape:MPSShape(vec_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* v = [g placeholderWithShape:MPSShape(vec_shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* gm = [g placeholderWithShape:MPSShape(vec_shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* eps =
            [g constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
        MPSGraphTensor* shifted = [g additionWithPrimaryTensor:v
                                              secondaryTensor:eps
                                                         name:nil];
        MPSGraphTensor* inv = [g reverseSquareRootWithTensor:shifted name:nil];
        NSArray<NSNumber*>* rest = @[ @0, @1, @2 ];

        // db = sum of the incoming gradient over everything but the channel.
        MPSGraphTensor* db_full = [g reductionSumWithTensor:dy
                                                       axes:rest
                                                       name:nil];
        MPSGraphTensor* db_v = [g reshapeTensor:db_full
                                      withShape:MPSShape(vec_shape)
                                           name:nil];
        // The other shared reduction: the gradient against the centred input.
        MPSGraphTensor* centred =
            [g subtractionWithPrimaryTensor:x
                            secondaryTensor:OverChannels(g, m, depth)
                                       name:nil];
        MPSGraphTensor* s2_full = [g
            reductionSumWithTensor:[g multiplicationWithPrimaryTensor:dy
                                                     secondaryTensor:centred
                                                                name:nil]
                              axes:rest
                              name:nil];
        MPSGraphTensor* s2 = [g reshapeTensor:s2_full
                                    withShape:MPSShape(vec_shape)
                                         name:nil];

        MPSGraphTensor* scaled_inv =
            scale_after ? [g multiplicationWithPrimaryTensor:inv
                                            secondaryTensor:gm
                                                       name:nil]
                        : inv;
        MPSGraphTensor* dx_t = [g
            multiplicationWithPrimaryTensor:dy
                            secondaryTensor:OverChannels(g, scaled_inv, depth)
                                       name:nil];
        MPSGraphTensor* dm_t =
            [g negativeWithTensor:[g multiplicationWithPrimaryTensor:db_v
                                                    secondaryTensor:scaled_inv
                                                               name:nil]
                             name:nil];
        // Gamma is not learned when the scale is not applied, so its gradient
        // is defined to be zero rather than left undefined.
        MPSGraphTensor* dg_t =
            scale_after ? [g multiplicationWithPrimaryTensor:s2
                                            secondaryTensor:inv
                                                       name:nil]
                        : [g constantWithScalar:0.0
                                          shape:MPSShape(vec_shape)
                                       dataType:MPSDataTypeFloat32];
        // -1/2 * (v + epsilon)^(-3/2), written as the CPU kernel writes it.
        MPSGraphTensor* half =
            [g constantWithScalar:-0.5 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* cube = [g
            divisionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:inv
                                                        secondaryTensor:half
                                                                   name:nil]
                      secondaryTensor:shifted
                                 name:nil];
        MPSGraphTensor* dv_t =
            [g multiplicationWithPrimaryTensor:s2
                               secondaryTensor:
                                   (scale_after
                                        ? [g multiplicationWithPrimaryTensor:cube
                                                             secondaryTensor:gm
                                                                        name:nil]
                                        : cube)
                                          name:nil];

        [out->inputs addObject:x];
        [out->inputs addObject:m];
        [out->inputs addObject:v];
        [out->inputs addObject:gm];
        [out->inputs addObject:dy];
        [out->outputs addObject:dx_t];
        [out->outputs addObject:dm_t];
        [out->outputs addObject:dv_t];
        [out->outputs addObject:db_v];
        [out->outputs addObject:dg_t];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* m_data =
      TensorDataForTensor(mean.get(), TF_FLOAT, device, status);
  if (m_data == nil) return;
  MPSGraphTensorData* v_data =
      TensorDataForTensor(variance.get(), TF_FLOAT, device, status);
  if (v_data == nil) return;
  MPSGraphTensorData* g_data =
      TensorDataForTensor(gamma.get(), TF_FLOAT, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* dy_data =
      TensorDataForTensor(backprop.get(), TF_FLOAT, device, status);
  if (dy_data == nil) return;

  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray array];
  ScopedTensor* outs[5] = {&dx, &dm, &dv, &db, &dg};
  for (int i = 0; i < 5; ++i) {
    MPSGraphTensorData* data =
        TensorDataForTensor(outs[i]->get(), TF_FLOAT, device, status);
    if (data == nil) return;
    [results addObject:data];
  }
  RunGraph(stream, *cached, @[ x_data, m_data, v_data, g_data, dy_data ],
           results, status);
}

#define METAL_BN_COMPUTE(NAME, IMPL)                                        \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<BatchNormOp*>(kernel);                           \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a batch normalisation kernel has no state.");    \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_BN_COMPUTE(BatchNorm_Compute, BatchNorm_ComputeImpl)
METAL_BN_COMPUTE(BatchNormGrad_Compute, BatchNormGrad_ComputeImpl)

#undef METAL_BN_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &BatchNormOp_Create, compute,
      &BatchNormOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
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

void RegisterMetalBatchNormGlobalKernels() {
  Register("BatchNormWithGlobalNormalization", &BatchNorm_Compute,
           "MetalBatchNormWithGlobalNormalization");
  Register("BatchNormWithGlobalNormalizationGrad", &BatchNormGrad_Compute,
           "MetalBatchNormWithGlobalNormalizationGrad");
}

}  // namespace metal
}  // namespace tensorflow
