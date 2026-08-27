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

// TensorFlow's quantiser does not round: it computes floor(x + 0.5) on a
// value it has already made non-negative. Mirroring the expression rather
// than substituting a rounding op keeps the boundary behaviour exact, which
// matters because the quantiser lands on exact halves constantly.
MPSGraphTensor* RoundHalfUp(MPSGraph* g, MPSGraphTensor* t) {
  MPSGraphTensor* half =
      [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
  return [g floorWithTensor:[g additionWithPrimaryTensor:t
                                         secondaryTensor:half
                                                    name:nil]
                       name:nil];
}

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
        MPSGraphTensor* steps =
            RoundHalfUp(g, [g multiplicationWithPrimaryTensor:shifted
                                             secondaryTensor:inv_scale
                                                        name:nil]);
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

/*** THE VARS FORMS ***/

// Same arithmetic as the Args forms, but min and max arrive as device
// tensors. Nudging needs them on the host, so this drains the stream once per
// call. That cost is real and is why the Args forms exist; quantisation-aware
// training uses the Vars forms during calibration, where a drain per step is
// tolerable and silently falling back to the host would be slower still.
bool ReadRangeOnHost(TF_OpKernelContext* ctx, SP_Stream stream, int min_index,
                     int max_index, float* min_value, float* max_value,
                     TF_Status* status) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
  ScopedTensor lo, hi;
  TF_GetInput(ctx, min_index, lo.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  TF_GetInput(ctx, max_index, hi.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const float* lo_p = static_cast<const float*>(TF_TensorData(lo.get()));
  const float* hi_p = static_cast<const float*>(TF_TensorData(hi.get()));
  if (lo_p == nullptr || hi_p == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FakeQuant range has no data.");
    return false;
  }
  *min_value = *lo_p;
  *max_value = *hi_p;
  return true;
}

void FakeQuantVars_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
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

  float min_value = 0.0f, max_value = 0.0f;
  if (!ReadRangeOnHost(ctx, stream, 1, 2, &min_value, &max_value, status)) {
    return;
  }
  float nudged_min, nudged_max, scale;
  Nudge(min_value, max_value, op->num_bits, op->narrow_range, &nudged_min,
        &nudged_max, &scale);

  std::string key = "FakeQuantVars";
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
        MPSGraphTensor* inv =
            [g constantWithScalar:1.0 / scale dataType:MPSDataTypeFloat32];
        MPSGraphTensor* sc =
            [g constantWithScalar:scale dataType:MPSDataTypeFloat32];
        MPSGraphTensor* clamped =
            [g clampWithTensor:x minValueTensor:lo maxValueTensor:hi name:nil];
        MPSGraphTensor* shifted =
            [g subtractionWithPrimaryTensor:clamped secondaryTensor:lo name:nil];
        MPSGraphTensor* steps =
            RoundHalfUp(g, [g multiplicationWithPrimaryTensor:shifted
                                             secondaryTensor:inv
                                                        name:nil]);
        [out->inputs addObject:x];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:
                             [g multiplicationWithPrimaryTensor:steps
                                                secondaryTensor:sc
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

// The Vars gradient produces three outputs: the pass-through gradient and the
// sums of the gradient below and above the nudged range, which are what the
// range parameters learn from.
void FakeQuantVarsGrad_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
                                   TF_Status* status) {
  ScopedTensor grad, input;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  ScopedTensor dx, dmin, dmax;
  dx.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  dmin.reset(TF_AllocateOutput(ctx, 1, TF_FLOAT, nullptr, 0,
                               TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  dmax.reset(TF_AllocateOutput(ctx, 2, TF_FLOAT, nullptr, 0,
                               TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  float min_value = 0.0f, max_value = 0.0f;
  if (!ReadRangeOnHost(ctx, stream, 2, 3, &min_value, &max_value, status)) {
    return;
  }
  float nudged_min, nudged_max, scale;
  Nudge(min_value, max_value, op->num_bits, op->narrow_range, &nudged_min,
        &nudged_max, &scale);

  std::string key = "FakeQuantVarsGrad";
  AppendShapeToKey(shape, &key);
  key.append("/n").append(std::to_string(nudged_min)).push_back(',');
  key.append(std::to_string(nudged_max));

  NSMutableArray<NSNumber*>* all_axes = [NSMutableArray array];
  for (size_t i = 0; i < shape.size(); ++i) {
    [all_axes addObject:@(static_cast<NSInteger>(i))];
  }
  const std::vector<int64_t> scalar_shape = {};

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
        MPSGraphTensor* below =
            [g lessThanWithPrimaryTensor:x secondaryTensor:lo name:nil];
        MPSGraphTensor* above =
            [g greaterThanWithPrimaryTensor:x secondaryTensor:hi name:nil];
        MPSGraphTensor* inside = [g
            logicalANDWithPrimaryTensor:[g notWithTensor:below name:nil]
                        secondaryTensor:[g notWithTensor:above name:nil]
                                   name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:x];
        [out->outputs addObject:[g selectWithPredicateTensor:inside
                                        truePredicateTensor:dy
                                       falsePredicateTensor:zero
                                                       name:nil]];
        [out->outputs
            addObject:[g reshapeTensor:
                             [g reductionSumWithTensor:
                                    [g selectWithPredicateTensor:below
                                                truePredicateTensor:dy
                                               falsePredicateTensor:zero
                                                               name:nil]
                                                  axes:all_axes
                                                  name:nil]
                             withShape:MPSShape(scalar_shape)
                                  name:nil]];
        [out->outputs
            addObject:[g reshapeTensor:
                             [g reductionSumWithTensor:
                                    [g selectWithPredicateTensor:above
                                                truePredicateTensor:dy
                                               falsePredicateTensor:zero
                                                               name:nil]
                                                  axes:all_axes
                                                  name:nil]
                             withShape:MPSShape(scalar_shape)
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
  MPSGraphTensorData* dx_data =
      TensorDataForTensor(dx.get(), TF_FLOAT, device, status);
  if (dx_data == nil) return;
  MPSGraphTensorData* dmin_data =
      TensorDataForTensor(dmin.get(), TF_FLOAT, device, status);
  if (dmin_data == nil) return;
  MPSGraphTensorData* dmax_data =
      TensorDataForTensor(dmax.get(), TF_FLOAT, device, status);
  if (dmax_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, x_data ],
           @[ dx_data, dmin_data, dmax_data ], status);
}

/*** THE PER-CHANNEL FORMS ***/

// One range per channel, so min and max are vectors rather than scalars.
//
// The nudge stays on the host, as in the scalar Vars forms, and for a sharper
// reason than symmetry with them. Nudging divides twice, and its result is
// then rounded; MPSGraph divides by a graph constant through a reciprocal
// multiply, which lands one unit in the last place away from an IEEE division.
// That is invisible almost everywhere and decisive here: an eight-bit
// symmetric range such as [-3, 3] puts the zero point at exactly 127.5, so one
// unit in the last place decides which side of the rounding boundary it falls
// on, and the whole channel comes out shifted by a quantisation step. Ranges
// symmetric about zero are the common case in quantisation-aware training, so
// this is not a corner. Doing the nudge on the host makes it the same
// arithmetic TensorFlow performs, by construction.
//
// The price is one stream drain per call, which is what the scalar Vars forms
// already pay and is documented alongside them.

// Reshapes a per-channel vector so it broadcasts against a rank-n input.
MPSGraphTensor* BroadcastChannels(MPSGraph* g, MPSGraphTensor* v, size_t rank,
                                  int64_t channels) {
  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  for (size_t i = 0; i + 1 < rank; ++i) [shape addObject:@1];
  [shape addObject:@(static_cast<NSInteger>(channels))];
  return [g reshapeTensor:v withShape:shape name:nil];
}

// Waits for everything already enqueued, then hands back the two range
// vectors' host-visible storage. The wait is what makes reading them legal.
bool ReadRangesOnHost(TF_OpKernelContext* ctx, SP_Stream stream, int min_index,
                      int max_index, int64_t channels, ScopedTensor* lo,
                      ScopedTensor* hi, const float** lo_data,
                      const float** hi_data, TF_Status* status) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
  TF_GetInput(ctx, min_index, lo->address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  TF_GetInput(ctx, max_index, hi->address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (NumElements(lo->get()) != channels ||
      NumElements(hi->get()) != channels) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the per-channel FakeQuant range must have one entry "
                 "per channel.");
    return false;
  }
  *lo_data = static_cast<const float*>(TF_TensorData(lo->get()));
  *hi_data = static_cast<const float*>(TF_TensorData(hi->get()));
  if (*lo_data == nullptr || *hi_data == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FakeQuant range has no data.");
    return false;
  }
  return true;
}

// Puts a host-computed per-channel vector where the graph can read it.
//
// The staging tensor is device memory that the host can write directly, which
// is what unified memory buys: no blit, no staging buffer. It is freed when
// the kernel returns, which is safe because this backend keeps one ordered
// stream, so anything that could reuse the storage is enqueued behind the work
// that reads it.
bool StageVector(TF_OpKernelContext* ctx, const std::vector<float>& values,
                 ScopedTensor* out, TF_Status* status) {
  int64_t dims[1] = {static_cast<int64_t>(values.size())};
  out->reset(TF_AllocateTemp(ctx, TF_FLOAT, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out->get());
  if (data == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: a staged per-channel vector has no storage.");
    return false;
  }
  std::memcpy(data, values.data(), values.size() * sizeof(float));
  return true;
}

// Nudges every channel on the host, then stages the three resulting vectors.
bool NudgePerChannel(TF_OpKernelContext* ctx, SP_Stream stream, QuantOp* op,
                     int min_index, int max_index, int64_t channels,
                     ScopedTensor* nudged_min, ScopedTensor* nudged_max,
                     ScopedTensor* scales, TF_Status* status) {
  ScopedTensor lo, hi;
  const float* lo_data = nullptr;
  const float* hi_data = nullptr;
  if (!ReadRangesOnHost(ctx, stream, min_index, max_index, channels, &lo, &hi,
                        &lo_data, &hi_data, status)) {
    return false;
  }
  std::vector<float> nlo(channels), nhi(channels), sc(channels);
  for (int64_t c = 0; c < channels; ++c) {
    Nudge(lo_data[c], hi_data[c], op->num_bits, op->narrow_range, &nlo[c],
          &nhi[c], &sc[c]);
  }
  return StageVector(ctx, nlo, nudged_min, status) &&
         StageVector(ctx, nhi, nudged_max, status) &&
         StageVector(ctx, sc, scales, status);
}

void FakeQuantPerChannel_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the per-channel FakeQuant needs a rank of at least "
                 "one.");
    return;
  }
  const int64_t channels = shape.back();
  const std::vector<int64_t> range_shape = {channels};
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

  ScopedTensor nudged_min, nudged_max, scales;
  if (!NudgePerChannel(ctx, stream, op, 1, 2, channels, &nudged_min,
                       &nudged_max, &scales, status)) {
    return;
  }

  std::string key = "FakeQuantPerChannel";
  AppendShapeToKey(shape, &key);

  const size_t rank = shape.size();
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* lo_v = [g placeholderWithShape:MPSShape(range_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* hi_v = [g placeholderWithShape:MPSShape(range_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* sc_v = [g placeholderWithShape:MPSShape(range_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* nlo = BroadcastChannels(g, lo_v, rank, channels);
        MPSGraphTensor* nhi = BroadcastChannels(g, hi_v, rank, channels);
        MPSGraphTensor* sc = BroadcastChannels(g, sc_v, rank, channels);
        MPSGraphTensor* clamped = [g clampWithTensor:x
                                      minValueTensor:nlo
                                      maxValueTensor:nhi
                                                name:nil];
        MPSGraphTensor* steps = RoundHalfUp(
            g, [g divisionWithPrimaryTensor:
                       [g subtractionWithPrimaryTensor:clamped
                                       secondaryTensor:nlo
                                                  name:nil]
                             secondaryTensor:sc
                                        name:nil]);
        [out->inputs addObject:x];
        [out->inputs addObject:lo_v];
        [out->inputs addObject:hi_v];
        [out->inputs addObject:sc_v];
        [out->outputs
            addObject:[g additionWithPrimaryTensor:
                             [g multiplicationWithPrimaryTensor:steps
                                                secondaryTensor:sc
                                                           name:nil]
                                   secondaryTensor:nlo
                                              name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* lo_data =
      TensorDataForTensor(nudged_min.get(), TF_FLOAT, device, status);
  if (lo_data == nil) return;
  MPSGraphTensorData* hi_data =
      TensorDataForTensor(nudged_max.get(), TF_FLOAT, device, status);
  if (hi_data == nil) return;
  MPSGraphTensorData* sc_data =
      TensorDataForTensor(scales.get(), TF_FLOAT, device, status);
  if (sc_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, lo_data, hi_data, sc_data ],
           @[ o_data ], status);
}

// Three outputs, as in the scalar form, except that the two range gradients
// are per-channel sums rather than scalars: everything but the last axis is
// reduced away.
void FakeQuantPerChannelGrad_ComputeImpl(QuantOp* op, TF_OpKernelContext* ctx,
                                         TF_Status* status) {
  ScopedTensor grad, input;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the per-channel FakeQuant gradient needs a rank of "
                 "at least one.");
    return;
  }
  const int64_t channels = shape.back();
  const std::vector<int64_t> range_shape = {channels};
  const int64_t count = ElementCount(shape);

  ScopedTensor dx, dmin, dmax;
  dx.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  dmin.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, range_shape.data(), 1,
      static_cast<size_t>(channels) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  dmax.reset(TF_AllocateOutput(
      ctx, 2, TF_FLOAT, range_shape.data(), 1,
      static_cast<size_t>(channels) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  ScopedTensor nudged_min, nudged_max, scales;
  if (!NudgePerChannel(ctx, stream, op, 2, 3, channels, &nudged_min,
                       &nudged_max, &scales, status)) {
    return;
  }

  std::string key = "FakeQuantPerChannelGrad";
  AppendShapeToKey(shape, &key);

  const size_t rank = shape.size();
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
        MPSGraphTensor* lo_v = [g placeholderWithShape:MPSShape(range_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* hi_v = [g placeholderWithShape:MPSShape(range_shape)
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        MPSGraphTensor* nlo = BroadcastChannels(g, lo_v, rank, channels);
        MPSGraphTensor* nhi = BroadcastChannels(g, hi_v, rank, channels);
        MPSGraphTensor* zero =
            [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        // Inclusive on both bounds, as in the scalar form: a value sitting
        // exactly on a bound passes its gradient through rather than
        // contributing to that bound.
        MPSGraphTensor* below = [g lessThanWithPrimaryTensor:x
                                            secondaryTensor:nlo
                                                       name:nil];
        MPSGraphTensor* above = [g greaterThanWithPrimaryTensor:x
                                               secondaryTensor:nhi
                                                          name:nil];
        MPSGraphTensor* inside = [g
            logicalANDWithPrimaryTensor:
                [g greaterThanOrEqualToWithPrimaryTensor:x
                                         secondaryTensor:nlo
                                                    name:nil]
                        secondaryTensor:
                            [g lessThanOrEqualToWithPrimaryTensor:x
                                                  secondaryTensor:nhi
                                                             name:nil]
                                   name:nil];
        // Everything but the channel axis is summed away.
        NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
        for (size_t i = 0; i + 1 < rank; ++i) {
          [axes addObject:@(static_cast<NSInteger>(i))];
        }
        MPSGraphTensor* dlo =
            [g selectWithPredicateTensor:below
                     truePredicateTensor:dy
                    falsePredicateTensor:zero
                                    name:nil];
        MPSGraphTensor* dhi =
            [g selectWithPredicateTensor:above
                     truePredicateTensor:dy
                    falsePredicateTensor:zero
                                    name:nil];
        if (rank > 1) {
          dlo = [g reductionSumWithTensor:dlo axes:axes name:nil];
          dhi = [g reductionSumWithTensor:dhi axes:axes name:nil];
        }
        [out->inputs addObject:dy];
        [out->inputs addObject:x];
        [out->inputs addObject:lo_v];
        [out->inputs addObject:hi_v];
        [out->outputs addObject:[g selectWithPredicateTensor:inside
                                         truePredicateTensor:dy
                                        falsePredicateTensor:zero
                                                        name:nil]];
        [out->outputs addObject:[g reshapeTensor:dlo
                                       withShape:MPSShape(range_shape)
                                            name:nil]];
        [out->outputs addObject:[g reshapeTensor:dhi
                                       withShape:MPSShape(range_shape)
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
  MPSGraphTensorData* lo_data =
      TensorDataForTensor(nudged_min.get(), TF_FLOAT, device, status);
  if (lo_data == nil) return;
  MPSGraphTensorData* hi_data =
      TensorDataForTensor(nudged_max.get(), TF_FLOAT, device, status);
  if (hi_data == nil) return;
  MPSGraphTensorData* dx_data =
      TensorDataForTensor(dx.get(), TF_FLOAT, device, status);
  if (dx_data == nil) return;
  MPSGraphTensorData* dmin_data =
      TensorDataForTensor(dmin.get(), TF_FLOAT, device, status);
  if (dmin_data == nil) return;
  MPSGraphTensorData* dmax_data =
      TensorDataForTensor(dmax.get(), TF_FLOAT, device, status);
  if (dmax_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, x_data, lo_data, hi_data ],
           @[ dx_data, dmin_data, dmax_data ], status);
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
METAL_COMPUTE(FakeQuantVars_Compute, FakeQuantVars_ComputeImpl)
METAL_COMPUTE(FakeQuantVarsGrad_Compute, FakeQuantVarsGrad_ComputeImpl)
METAL_COMPUTE(FakeQuantPerChannel_Compute, FakeQuantPerChannel_ComputeImpl)
METAL_COMPUTE(FakeQuantPerChannelGrad_Compute,
              FakeQuantPerChannelGrad_ComputeImpl)

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
  // The Args forms take their bounds as attributes; the Vars forms take them
  // as device tensors and pay a stream drain per call to nudge on the host.
  Register("FakeQuantWithMinMaxArgs", &FakeQuant_Compute,
           "MetalFakeQuantWithMinMaxArgs");
  Register("FakeQuantWithMinMaxArgsGradient", &FakeQuantGrad_Compute,
           "MetalFakeQuantWithMinMaxArgsGradient");
  Register("FakeQuantWithMinMaxVars", &FakeQuantVars_Compute,
           "MetalFakeQuantWithMinMaxVars");
  Register("FakeQuantWithMinMaxVarsGradient", &FakeQuantVarsGrad_Compute,
           "MetalFakeQuantWithMinMaxVarsGradient");
  Register("FakeQuantWithMinMaxVarsPerChannel", &FakeQuantPerChannel_Compute,
           "MetalFakeQuantWithMinMaxVarsPerChannel");
  Register("FakeQuantWithMinMaxVarsPerChannelGradient",
           &FakeQuantPerChannelGrad_Compute,
           "MetalFakeQuantWithMinMaxVarsPerChannelGradient");
}

}  // namespace metal
}  // namespace tensorflow
