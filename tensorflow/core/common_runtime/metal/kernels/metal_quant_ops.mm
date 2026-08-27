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
#include <cmath>
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

// FakeQuantWithMinMaxArgs and its gradient.
//
// The quantisation is simple arithmetic, but the nudging is not, and getting
// it wrong is invisible: the output still looks like a quantised tensor, just
// one whose zero point is off by a fraction of a step. TensorFlow's rule,
// which this reproduces, is to first widen [min, max] so that zero is
// representable, then round the resulting zero point to an integer and derive
// the nudged bounds from it. Copying the arithmetic without that step gives
// results that differ from the CPU kernel in the last bits everywhere and by
// a whole step near zero.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct QuantOp {
  float min_value = -6.0f;
  float max_value = 6.0f;
  int num_bits = 8;
  bool narrow_range = false;
};

void* QuantOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new QuantOp();
  TF_OpKernelConstruction_GetAttrFloat(ctx, "min", &op->min_value, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "max", &op->max_value, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  int32_t bits = 8;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_bits", &bits, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->num_bits = bits;
  TF_Bool narrow = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "narrow_range", &narrow, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->narrow_range = narrow != 0;

  if (op->num_bits < 2 || op->num_bits > 16) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FakeQuant num_bits must be between 2 and 16.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void QuantOp_Delete(void* kernel) { delete static_cast<QuantOp*>(kernel); }

// TensorFlow's Nudge, reproduced exactly: the zero point is rounded to an
// integer and the bounds are recomputed from it, so that zero maps to a
// representable value.
void Nudge(float min_value, float max_value, int num_bits, bool narrow_range,
           float* nudged_min, float* nudged_max, float* scale) {
  const float quant_min = narrow_range ? 1.0f : 0.0f;
  const float quant_max = static_cast<float>((1 << num_bits) - 1);
  *scale = (max_value - min_value) / (quant_max - quant_min);
  const float zero_point_from_min =
      quant_min - min_value / (*scale == 0.0f ? 1.0f : *scale);
  float nudged_zero_point;
  if (zero_point_from_min < quant_min) {
    nudged_zero_point = quant_min;
  } else if (zero_point_from_min > quant_max) {
    nudged_zero_point = quant_max;
  } else {
    nudged_zero_point = std::round(zero_point_from_min);
  }
  *nudged_min = (quant_min - nudged_zero_point) * (*scale);
  *nudged_max = (quant_max - nudged_zero_point) * (*scale);
}

void FakeQuant_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  float nudged_min, nudged_max, scale;
  Nudge(op->min_value, op->max_value, op->num_bits, op->narrow_range,
        &nudged_min, &nudged_max, &scale);

  std::string key = "FakeQuantArgs";
  AppendShapeToKey(shape, &key);
  key.append("/n").append(std::to_string(nudged_min)).push_back(',');
  key.append(std::to_string(nudged_max)).push_back(',');
  key.append(std::to_string(scale));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* lo =
            [g constantWithScalar:nudged_min dataType:MPSDataTypeFloat32];
        MPSGraphTensor* hi =
            [g constantWithScalar:nudged_max dataType:MPSDataTypeFloat32];
        MPSGraphTensor* inv_scale =
            [g constantWithScalar:1.0 / scale dataType:MPSDataTypeFloat32];
        MPSGraphTensor* s =
            [g constantWithScalar:scale dataType:MPSDataTypeFloat32];
        MPSGraphTensor* clamped =
            [g clampWithTensor:x minValueTensor:lo maxValueTensor:hi name:nil];
        MPSGraphTensor* shifted =
            [g subtractionWithPrimaryTensor:clamped secondaryTensor:lo name:nil];
        // TensorFlow rounds half away from zero here, which is round, not rint.
        MPSGraphTensor* steps = [g roundWithTensor:
                                       [g multiplicationWithPrimaryTensor:shifted
                                                          secondaryTensor:inv_scale
                                                                     name:nil]
                                              name:nil];
        [out->inputs addObject:x];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:
                             [g multiplicationWithPrimaryTensor:steps
                                                secondaryTensor:s
                                                           name:nil]
                                   secondaryTensor:lo
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ o_data ], status);
}

// The gradient passes through inside the nudged range and is zero outside it.
void FakeQuantGrad_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor grad, input;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  float nudged_min, nudged_max, scale;
  Nudge(op->min_value, op->max_value, op->num_bits, op->narrow_range,
        &nudged_min, &nudged_max, &scale);

  std::string key = "FakeQuantArgsGrad";
  AppendShapeToKey(shape, &key);
  key.append("/n").append(std::to_string(nudged_min)).push_back(',');
  key.append(std::to_string(nudged_max));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(shape)
                                            dataType:MPSDataTypeFloat32
                                                name:nil];
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* lo =
            [g constantWithScalar:nudged_min dataType:MPSDataTypeFloat32];
        MPSGraphTensor* hi =
            [g constantWithScalar:nudged_max dataType:MPSDataTypeFloat32];
        MPSGraphTensor* zero =
            [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        // Inclusive on both bounds, which is what TensorFlow's kernel does.
        MPSGraphTensor* above =
            [g greaterThanOrEqualToWithPrimaryTensor:x
                                     secondaryTensor:lo
                                                name:nil];
        MPSGraphTensor* below =
            [g lessThanOrEqualToWithPrimaryTensor:x
                                  secondaryTensor:hi
                                             name:nil];
        MPSGraphTensor* inside =
            [g logicalANDWithPrimaryTensor:above secondaryTensor:below name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:x];
        [out->outputs addObject:[g selectWithPredicateTensor:inside
                                        truePredicateTensor:dy
                                       falsePredicateTensor:zero
                                                       name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), TF_FLOAT, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, x_data ], @[ o_data ], status);
}

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<QuantOp*>(kernel);                                 \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(FakeQuant_Compute, FakeQuant_ComputeImpl)
METAL_COMPUTE(FakeQuantGrad_Compute, FakeQuantGrad_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &QuantOp_Create, compute, &QuantOp_Delete);
  TF_RegisterKernelBuilder(name.c_str(), builder, status);
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalQuantKernels() {
  // Only the Args forms, whose bounds are attributes. The Vars forms take the
  // range as device tensors, which would mean nudging on the host and draining
  // the stream on every call.
  Register("FakeQuantWithMinMaxArgs", &FakeQuant_Compute,
           "MetalFakeQuantWithMinMaxArgs");
  Register("FakeQuantWithMinMaxArgsGradient", &FakeQuantGrad_Compute,
           "MetalFakeQuantWithMinMaxArgsGradient");
}

}  // namespace metal
}  // namespace tensorflow
