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
#include <limits>
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

// QuantizeAndDequantize and its V2, V3 and V4 forms, plus the V4 gradient.
//
// The arithmetic that picks the scale is a handful of scalars per channel, and
// it is written on the host for the same reason the FakeQuant range is: it
// divides and then rounds, and reproducing TensorFlow's result on a rounding
// boundary means performing TensorFlow's divisions. The elementwise part, which
// is the whole tensor, stays on the GPU.
//
// The range inputs are pinned to host memory, so the common case where the
// caller supplies the range costs no synchronisation at all. When the range is
// not given it has to be measured, which is a reduction over the input and
// therefore one drain; TensorFlow's own kernel copies the same two values back
// to the host at the same point.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct QuantizeDequantizeOp {
  bool signed_input = true;
  int num_bits = 8;
  bool range_given = false;
  bool narrow_range = false;
  bool round_half_up = false;
  int axis = -1;
  // Only the deprecated V1 form carries its range as attributes.
  bool range_from_attrs = false;
  float attr_min = 0.0f;
  float attr_max = 0.0f;
};

void* QuantizeDequantizeOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new QuantizeDequantizeOp();

  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "signed_input", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->signed_input = flag != 0;
  TF_SetStatus(status, TF_OK, "");

  int32_t bits = 8;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_bits", &bits, status);
  if (TF_GetCode(status) == TF_OK) op->num_bits = bits;
  TF_SetStatus(status, TF_OK, "");

  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "range_given", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->range_given = flag != 0;
  TF_SetStatus(status, TF_OK, "");

  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "narrow_range", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->narrow_range = flag != 0;
  TF_SetStatus(status, TF_OK, "");

  int32_t axis = -1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "axis", &axis, status);
  if (TF_GetCode(status) == TF_OK) op->axis = axis;
  TF_SetStatus(status, TF_OK, "");

  // V3 has no round_mode at all and always rounds halves to even.
  char mode[24] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "round_mode", mode,
                                        sizeof(mode) - 1, status);
  if (TF_GetCode(status) == TF_OK && std::strcmp(mode, "HALF_UP") == 0) {
    op->round_half_up = true;
  }
  TF_SetStatus(status, TF_OK, "");

  // The deprecated form takes its range as two float attributes.
  float value = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "input_min", &value, status);
  if (TF_GetCode(status) == TF_OK) {
    op->range_from_attrs = true;
    op->attr_min = value;
    TF_OpKernelConstruction_GetAttrFloat(ctx, "input_max", &value, status);
    if (TF_GetCode(status) == TF_OK) op->attr_max = value;
  }
  TF_SetStatus(status, TF_OK, "");

  TF_DeleteStatus(status);
  return op;
}

void QuantizeDequantizeOp_Delete(void* kernel) {
  delete static_cast<QuantizeDequantizeOp*>(kernel);
}

// TensorFlow's ComputeQuantizationRange, reproduced exactly. It both picks the
// scale and moves whichever end of the range did not determine it, so the two
// have to be computed together.
void ComputeQuantizationRange(bool signed_input, int num_bits,
                              bool narrow_range, float* min_range,
                              float* max_range, float* scale,
                              float* inverse_scale) {
  const int64_t min_quantized =
      signed_input ? (narrow_range ? -(1LL << (num_bits - 1)) + 1
                                   : -(1LL << (num_bits - 1)))
                   : 0;
  const int64_t max_quantized =
      signed_input ? (1LL << (num_bits - 1)) - 1 : (1LL << num_bits) - 1;
  const float scale_from_min_side =
      (min_quantized * *min_range > 0)
          ? static_cast<float>(min_quantized) / *min_range
          : std::numeric_limits<float>::max();
  const float scale_from_max_side =
      (max_quantized * *max_range > 0)
          ? static_cast<float>(max_quantized) / *max_range
          : std::numeric_limits<float>::max();
  // Avoids changing the side of the range that determined the scale.
  if (scale_from_min_side < scale_from_max_side) {
    *scale = scale_from_min_side;
    *inverse_scale = *min_range / static_cast<float>(min_quantized);
    *max_range = static_cast<float>(max_quantized) * *inverse_scale;
  } else {
    *scale = scale_from_max_side;
    *inverse_scale = *max_range / static_cast<float>(max_quantized);
    *min_range = static_cast<float>(min_quantized) * *inverse_scale;
  }
}

// Puts a host-computed per-channel vector where the graph can read it. Unified
// memory means the CPU can write device storage directly, with no staging
// buffer and no blit.
bool StageVector(TF_OpKernelContext* ctx, const std::vector<float>& values,
                 ScopedTensor* out, TF_Status* status) {
  int64_t dims[1] = {static_cast<int64_t>(values.size())};
  out->reset(TF_AllocateTemp(ctx, TF_FLOAT, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out->get());
  if (data == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: a staged range vector has no storage.");
    return false;
  }
  std::memcpy(data, values.data(), values.size() * sizeof(float));
  return true;
}

// A per-channel vector reshaped so it broadcasts against the input: all ones
// except at the quantisation axis.
NSArray<NSNumber*>* BroadcastShape(size_t rank, int axis, int64_t channels) {
  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  for (size_t i = 0; i < rank; ++i) {
    const bool here = axis >= 0 && static_cast<size_t>(axis) == i;
    [shape addObject:@(here ? static_cast<NSInteger>(channels) : 1)];
  }
  return shape;
}

// Measures the range on the device when the caller did not supply one, then
// reads it back. This is the one path that has to synchronise.
bool MeasureRange(TF_OpKernelContext* ctx, SP_Stream stream, id<MTLDevice> dev,
                  TF_Tensor* input, const std::vector<int64_t>& shape, int axis,
                  int64_t channels, std::vector<float>* mins,
                  std::vector<float>* maxes, TF_Status* status) {
  const std::vector<int64_t> range_shape = {channels};
  ScopedTensor lo, hi;
  int64_t dims[1] = {channels};
  lo.reset(TF_AllocateTemp(ctx, TF_FLOAT, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  hi.reset(TF_AllocateTemp(ctx, TF_FLOAT, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;

  std::string key = "QuantizeDequantizeRange";
  AppendShapeToKey(shape, &key);
  key.append("/a").append(std::to_string(axis));

  const size_t rank = shape.size();
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
        for (size_t i = 0; i < rank; ++i) {
          if (axis >= 0 && static_cast<size_t>(axis) == i) continue;
          [axes addObject:@(static_cast<NSInteger>(i))];
        }
        MPSGraphTensor* mn = [g reductionMinimumWithTensor:x
                                                      axes:axes
                                                      name:nil];
        MPSGraphTensor* mx = [g reductionMaximumWithTensor:x
                                                      axes:axes
                                                      name:nil];
        [out->inputs addObject:x];
        [out->outputs addObject:[g reshapeTensor:mn
                                       withShape:MPSShape(range_shape)
                                            name:nil]];
        [out->outputs addObject:[g reshapeTensor:mx
                                       withShape:MPSShape(range_shape)
                                            name:nil]];
      },
      status);
  if (cached == nullptr) return false;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input, TF_FLOAT, dev, status);
  if (x_data == nil) return false;
  MPSGraphTensorData* lo_data =
      TensorDataForTensor(lo.get(), TF_FLOAT, dev, status);
  if (lo_data == nil) return false;
  MPSGraphTensorData* hi_data =
      TensorDataForTensor(hi.get(), TF_FLOAT, dev, status);
  if (hi_data == nil) return false;
  if (!RunGraph(stream, *cached, @[ x_data ], @[ lo_data, hi_data ], status)) {
    return false;
  }

  WaitForStream(stream);
  const float* lo_p = static_cast<const float*>(TF_TensorData(lo.get()));
  const float* hi_p = static_cast<const float*>(TF_TensorData(hi.get()));
  if (lo_p == nullptr || hi_p == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: the measured range has no storage.");
    return false;
  }
  mins->assign(lo_p, lo_p + channels);
  maxes->assign(hi_p, hi_p + channels);
  return true;
}

// Reads a range that arrived as a host-memory input.
bool ReadGivenRange(TF_OpKernelContext* ctx, int min_index, int max_index,
                    int64_t channels, std::vector<float>* mins,
                    std::vector<float>* maxes, TF_Status* status) {
  ScopedTensor lo, hi;
  TF_GetInput(ctx, min_index, lo.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  TF_GetInput(ctx, max_index, hi.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const float* lo_p = static_cast<const float*>(TF_TensorData(lo.get()));
  const float* hi_p = static_cast<const float*>(TF_TensorData(hi.get()));
  if (lo_p == nullptr || hi_p == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the quantisation range has no data.");
    return false;
  }
  const int64_t given = NumElements(lo.get());
  if (given != channels && given != 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the quantisation range does not match the axis.");
    return false;
  }
  mins->assign(channels, lo_p[0]);
  maxes->assign(channels, hi_p[0]);
  if (given == channels) {
    mins->assign(lo_p, lo_p + channels);
    maxes->assign(hi_p, hi_p + channels);
  }
  return true;
}

void QuantizeDequantize_ComputeImpl(QuantizeDequantizeOp* op,
                                    TF_OpKernelContext* ctx, int min_index,
                                    int max_index, int num_bits_index,
                                    TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const size_t rank = shape.size();
  int axis = op->axis;
  if (axis >= static_cast<int>(rank)) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the quantisation axis is out of range.");
    return;
  }
  const int64_t channels = axis < 0 ? 1 : shape[axis];
  const int64_t count = ElementCount(shape);

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(rank),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  // V3 takes the bit width as an input rather than an attribute.
  int num_bits = op->num_bits;
  if (num_bits_index >= 0) {
    ScopedTensor bits;
    TF_GetInput(ctx, num_bits_index, bits.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    const void* data = TF_TensorData(bits.get());
    if (data == nullptr) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: num_bits has no data.");
      return;
    }
    num_bits = *static_cast<const int32_t*>(data);
  }
  if (num_bits < 2 || num_bits > 30) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: num_bits must lie between 2 and 30.");
    return;
  }

  std::vector<float> mins, maxes;
  if (op->range_from_attrs) {
    mins.assign(channels, op->attr_min);
    maxes.assign(channels, op->attr_max);
  } else if (op->range_given) {
    if (!ReadGivenRange(ctx, min_index, max_index, channels, &mins, &maxes,
                        status)) {
      return;
    }
  } else if (!MeasureRange(ctx, stream, device, input.get(), shape, axis,
                           channels, &mins, &maxes, status)) {
    return;
  }

  std::vector<float> scales(channels), inverses(channels);
  for (int64_t c = 0; c < channels; ++c) {
    ComputeQuantizationRange(op->signed_input, num_bits, op->narrow_range,
                             &mins[c], &maxes[c], &scales[c], &inverses[c]);
  }

  // The range is clamped only when it was given: when it was measured, the
  // input already lies inside it, and TensorFlow deliberately skips the clamp
  // there rather than clamping to a range it has just moved.
  const bool clamp = op->range_given || op->range_from_attrs;
  ScopedTensor lo_t, hi_t, scale_t, inv_t;
  if (clamp) {
    if (!StageVector(ctx, mins, &lo_t, status)) return;
    if (!StageVector(ctx, maxes, &hi_t, status)) return;
  }
  if (!StageVector(ctx, scales, &scale_t, status)) return;
  if (!StageVector(ctx, inverses, &inv_t, status)) return;

  std::string key = "QuantizeDequantize";
  AppendShapeToKey(shape, &key);
  key.append("/a").append(std::to_string(axis));
  key.append(clamp ? "/clamp" : "/free");
  key.append(op->round_half_up ? "/up" : "/even");

  const bool round_half_up = op->round_half_up;
  const std::vector<int64_t> range_shape = {channels};
  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        NSArray<NSNumber*>* broadcast = BroadcastShape(rank, axis, channels);
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(shape)
                                           dataType:MPSDataTypeFloat32
                                               name:nil];
        MPSGraphTensor* y = x;
        if (clamp) {
          MPSGraphTensor* lo_v =
              [g placeholderWithShape:MPSShape(range_shape)
                             dataType:MPSDataTypeFloat32
                                 name:nil];
          MPSGraphTensor* hi_v =
              [g placeholderWithShape:MPSShape(range_shape)
                             dataType:MPSDataTypeFloat32
                                 name:nil];
          [out->inputs addObject:x];
          [out->inputs addObject:lo_v];
          [out->inputs addObject:hi_v];
          y = [g clampWithTensor:x
              minValueTensor:[g reshapeTensor:lo_v
                                    withShape:broadcast
                                         name:nil]
              maxValueTensor:[g reshapeTensor:hi_v
                                    withShape:broadcast
                                         name:nil]
                        name:nil];
        } else {
          [out->inputs addObject:x];
        }
        MPSGraphTensor* scale_v = [g placeholderWithShape:MPSShape(range_shape)
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
        MPSGraphTensor* inv_v = [g placeholderWithShape:MPSShape(range_shape)
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        [out->inputs addObject:scale_v];
        [out->inputs addObject:inv_v];
        MPSGraphTensor* scaled = [g
            multiplicationWithPrimaryTensor:y
                            secondaryTensor:[g reshapeTensor:scale_v
                                                   withShape:broadcast
                                                        name:nil]
                                       name:nil];
        // HALF_UP is floor(x + 0.5); HALF_TO_EVEN is rint. MPSGraph's round
        // is neither: it rounds halves away from zero.
        MPSGraphTensor* rounded;
        if (round_half_up) {
          MPSGraphTensor* half =
              [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
          rounded = [g floorWithTensor:[g additionWithPrimaryTensor:scaled
                                                    secondaryTensor:half
                                                               name:nil]
                                  name:nil];
        } else {
          rounded = [g rintWithTensor:scaled name:nil];
        }
        [out->outputs
            addObject:[g multiplicationWithPrimaryTensor:rounded
                                         secondaryTensor:
                                             [g reshapeTensor:inv_v
                                                    withShape:broadcast
                                                         name:nil]
                                                    name:nil]];
      },
      status);
  if (cached == nullptr) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray array];
  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), TF_FLOAT, device, status);
  if (x_data == nil) return;
  [feeds addObject:x_data];
  if (clamp) {
    MPSGraphTensorData* lo_data =
        TensorDataForTensor(lo_t.get(), TF_FLOAT, device, status);
    if (lo_data == nil) return;
    MPSGraphTensorData* hi_data =
        TensorDataForTensor(hi_t.get(), TF_FLOAT, device, status);
    if (hi_data == nil) return;
    [feeds addObject:lo_data];
    [feeds addObject:hi_data];
  }
  MPSGraphTensorData* scale_data =
      TensorDataForTensor(scale_t.get(), TF_FLOAT, device, status);
  if (scale_data == nil) return;
  MPSGraphTensorData* inv_data =
      TensorDataForTensor(inv_t.get(), TF_FLOAT, device, status);
  if (inv_data == nil) return;
  [feeds addObject:scale_data];
  [feeds addObject:inv_data];

  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), TF_FLOAT, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, feeds, @[ o_data ], status);
}

// The V4 gradient passes the gradient through wherever the input lay inside
// the given range and drops it elsewhere. The two range gradients are defined
// to be zero, which is what TensorFlow emits as well.
void QuantizeDequantizeGrad_ComputeImpl(QuantizeDequantizeOp* op,
                                        TF_OpKernelContext* ctx,
                                        TF_Status* status) {
  ScopedTensor grad, input;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const size_t rank = shape.size();
  int axis = op->axis;
  if (axis >= static_cast<int>(rank)) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the quantisation axis is out of range.");
    return;
  }
  const int64_t channels = axis < 0 ? 1 : shape[axis];
  const int64_t count = ElementCount(shape);
  const std::vector<int64_t> range_shape = {channels};

  ScopedTensor dx, dmin, dmax;
  dx.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(rank),
      static_cast<size_t>(count) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  // The range gradients keep the shape of the range: a scalar without an axis.
  dmin.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, range_shape.data(), axis < 0 ? 0 : 1,
      static_cast<size_t>(channels) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  dmax.reset(TF_AllocateOutput(
      ctx, 2, TF_FLOAT, range_shape.data(), axis < 0 ? 0 : 1,
      static_cast<size_t>(channels) * TF_DataTypeSize(TF_FLOAT), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::vector<float> mins, maxes;
  if (!ReadGivenRange(ctx, 2, 3, channels, &mins, &maxes, status)) return;
  ScopedTensor lo_t, hi_t;
  if (!StageVector(ctx, mins, &lo_t, status)) return;
  if (!StageVector(ctx, maxes, &hi_t, status)) return;

  // Zeroing the two range gradients is a fill, not a graph result, so it can
  // be a blit rather than a second reduction.
  BufferSlice dmin_slice, dmax_slice;
  if (!SliceForTensor(dmin.get(), &dmin_slice, status)) return;
  if (!SliceForTensor(dmax.get(), &dmax_slice, status)) return;
  {
    OrderedCommandBuffer command_buffer(stream);
    if (!command_buffer.ok()) {
      TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                   "Metal: could not create a command buffer for the "
                   "quantisation gradient.");
      return;
    }
    id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
    const NSUInteger bytes =
        static_cast<NSUInteger>(channels) * sizeof(float);
    [zero fillBuffer:dmin_slice.buffer
               range:NSMakeRange(dmin_slice.offset, bytes)
               value:0];
    [zero fillBuffer:dmax_slice.buffer
               range:NSMakeRange(dmax_slice.offset, bytes)
               value:0];
    [zero endEncoding];
    command_buffer.Commit();
  }

  std::string key = "QuantizeDequantizeGrad";
  AppendShapeToKey(shape, &key);
  key.append("/a").append(std::to_string(axis));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        NSArray<NSNumber*>* broadcast = BroadcastShape(rank, axis, channels);
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
        MPSGraphTensor* lo = [g reshapeTensor:lo_v
                                    withShape:broadcast
                                         name:nil];
        MPSGraphTensor* hi = [g reshapeTensor:hi_v
                                    withShape:broadcast
                                         name:nil];
        MPSGraphTensor* zero =
            [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor* inside = [g
            logicalANDWithPrimaryTensor:
                [g greaterThanOrEqualToWithPrimaryTensor:x
                                         secondaryTensor:lo
                                                    name:nil]
                        secondaryTensor:
                            [g lessThanOrEqualToWithPrimaryTensor:x
                                                  secondaryTensor:hi
                                                             name:nil]
                                   name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:x];
        [out->inputs addObject:lo_v];
        [out->inputs addObject:hi_v];
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
  MPSGraphTensorData* lo_data =
      TensorDataForTensor(lo_t.get(), TF_FLOAT, device, status);
  if (lo_data == nil) return;
  MPSGraphTensorData* hi_data =
      TensorDataForTensor(hi_t.get(), TF_FLOAT, device, status);
  if (hi_data == nil) return;
  MPSGraphTensorData* dx_data =
      TensorDataForTensor(dx.get(), TF_FLOAT, device, status);
  if (dx_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, x_data, lo_data, hi_data ],
           @[ dx_data ], status);
}

#define METAL_QD_COMPUTE(NAME, BODY)                                        \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<QuantizeDequantizeOp*>(kernel);                  \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a quantisation kernel has no state.");           \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_QD_COMPUTE(QuantizeDequantizeV1_Compute,
                 QuantizeDequantize_ComputeImpl(op, ctx, -1, -1, -1, status))
METAL_QD_COMPUTE(QuantizeDequantizeV2_Compute,
                 QuantizeDequantize_ComputeImpl(op, ctx, 1, 2, -1, status))
METAL_QD_COMPUTE(QuantizeDequantizeV3_Compute,
                 QuantizeDequantize_ComputeImpl(op, ctx, 1, 2, 3, status))
METAL_QD_COMPUTE(QuantizeDequantizeGrad_Compute,
                 QuantizeDequantizeGrad_ComputeImpl(op, ctx, status))

#undef METAL_QD_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &QuantizeDequantizeOp_Create,
                          compute, &QuantizeDequantizeOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  for (const char* input : host_inputs) {
    TF_KernelBuilder_HostMemory(builder, input);
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

void RegisterMetalQuantizeDequantizeKernels() {
  // The range and the bit width are a handful of scalars that the scale
  // arithmetic needs on the host, so they are placed there and TensorFlow
  // inserts the transfer. That keeps the supplied-range case free of any
  // synchronisation.
  Register("QuantizeAndDequantize", &QuantizeDequantizeV1_Compute,
           "MetalQuantizeAndDequantize", {});
  Register("QuantizeAndDequantizeV2", &QuantizeDequantizeV2_Compute,
           "MetalQuantizeAndDequantizeV2", {"input_min", "input_max"});
  Register("QuantizeAndDequantizeV3", &QuantizeDequantizeV3_Compute,
           "MetalQuantizeAndDequantizeV3",
           {"input_min", "input_max", "num_bits"});
  Register("QuantizeAndDequantizeV4", &QuantizeDequantizeV2_Compute,
           "MetalQuantizeAndDequantizeV4", {"input_min", "input_max"});
  Register("QuantizeAndDequantizeV4Grad", &QuantizeDequantizeGrad_Compute,
           "MetalQuantizeAndDequantizeV4Grad", {"input_min", "input_max"});
}

}  // namespace metal
}  // namespace tensorflow
