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

// FusedBatchNorm and its gradient, for v1, v2 and v3.
//
// The arithmetic follows tensorflow/core/kernels/fused_batch_norm_op.cc
// exactly rather than a textbook description of batch normalisation, because
// two details differ between them and both are observable:
//
//   * The variance used to normalise, and handed to the gradient through
//     reserve_space_2, is the biased estimate (divided by N). The variance
//     returned as the batch_variance output, which is what feeds a moving
//     average, carries Bessel's correction (multiplied by N/(N-1)). Using one
//     where the other belongs does not fail, it slowly biases the statistics a
//     model is evaluated with.
//   * exponential_avg_factor, when it is not 1, blends the new statistics with
//     the old ones passed in as the mean and variance inputs.
//
// The gradient is likewise the form that file computes:
//   inv_std   = rsqrt(variance + epsilon)
//   x_center  = x - mean
//   dscale    = sum(dy * x_center * inv_std)
//   doffset   = sum(dy)
//   dx        = scale * inv_std *
//               (dy - doffset/N - x_center * inv_std^2 * sum(dy*x_center)/N)

struct BatchNormOp {
  // _FusedBatchNormEx folds an optional side input and an optional activation
  // into the same pass. Both are recorded here; everything else about the op
  // is unchanged.
  bool has_side_input = false;
  bool relu = false;
  TF_DataType dtype = TF_FLOAT;       // T, the data tensor
  TF_DataType param_dtype = TF_FLOAT;  // U, the statistics
  float epsilon = 1e-4f;
  float exponential_avg_factor = 1.0f;
  bool nhwc = true;
  bool is_training = true;
};

void* BatchNormOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BatchNormOp();

  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  // v1 has no U; the statistics are then the same type as the data.
  TF_OpKernelConstruction_GetAttrType(ctx, "U", &op->param_dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->param_dtype = op->dtype;
  }

  TF_OpKernelConstruction_GetAttrFloat(ctx, "epsilon", &op->epsilon, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrFloat(ctx, "exponential_avg_factor",
                                       &op->exponential_avg_factor, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->exponential_avg_factor = 1.0f;
  }

  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->nhwc = true;
  } else {
    op->nhwc = std::strcmp(format, "NCHW") != 0;
  }

  TF_Bool is_training = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_training", &is_training, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->is_training = is_training != 0;
  int32_t side_inputs = 0;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_side_inputs", &side_inputs,
                                       status);
  if (TF_GetCode(status) == TF_OK) op->has_side_input = side_inputs > 0;
  TF_SetStatus(status, TF_OK, "");
  char activation[24] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "activation_mode", activation,
                                        sizeof(activation) - 1, status);
  if (TF_GetCode(status) == TF_OK && activation[0] != '\0') {
    if (std::strcmp(activation, "Relu") == 0) {
      op->relu = true;
    } else if (std::strcmp(activation, "Identity") != 0) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: the fused batch normalisation supports the "
                   "Identity and Relu activations only.");
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
  }
  TF_SetStatus(status, TF_OK, "");

  TF_DeleteStatus(status);
  return op;
}

// A rank-0 output that exists only to fill a slot in the op's signature.
//
// Asking for zero bytes gets a scalar whose one element has no storage behind
// it, so whatever the allocator last left there is what the caller reads.
// reserve_space_3 came back as zero on a fresh allocation and as arbitrary
// values once the allocator had recycled memory, which is a graph output that
// changes with what ran before it. The slot is given its element and that
// element is set to zero, which is what TensorFlow's own kernels leave there.
bool AllocateEmptyScalar(TF_OpKernelContext* ctx, int index, TF_DataType dtype,
                         TF_Status* status) {
  ScopedTensor scalar;
  scalar.reset(TF_AllocateOutput(ctx, index, dtype, nullptr, 0,
                                 TF_DataTypeSize(dtype), status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(scalar.get());
  if (data != nullptr) {
    // Freshly allocated, with nothing in flight against it, so the host may
    // write it directly.
    std::memset(data, 0, TF_DataTypeSize(dtype));
  }
  return true;
}

void BatchNormOp_Delete(void* kernel) {
  delete static_cast<BatchNormOp*>(kernel);
}

// Axes reduced over: everything but the channel axis.
NSArray<NSNumber*>* ReduceAxes(bool nhwc) {
  return nhwc ? @[ @0, @1, @2 ] : @[ @0, @2, @3 ];
}

// Shape a per-channel vector takes inside the graph so it broadcasts against
// the data tensor under either layout.
std::vector<int64_t> ChannelShape(bool nhwc, int64_t channels) {
  return nhwc ? std::vector<int64_t>{1, 1, 1, channels}
              : std::vector<int64_t>{1, channels, 1, 1};
}

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

/*** FORWARD ***/

void FusedBatchNorm_ComputeImpl(BatchNormOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor x, scale, offset, in_mean, in_variance;
  TF_GetInput(ctx, 0, x.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, scale.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, offset.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, in_mean.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, in_variance.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor side_input;
  if (op->has_side_input) {
    TF_GetInput(ctx, 5, side_input.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  if (TF_NumDims(x.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FusedBatchNorm expects a rank-4 input.");
    return;
  }
  const std::vector<int64_t> x_shape = ShapeOf(x.get());
  const int64_t channels = op->nhwc ? x_shape[3] : x_shape[1];
  const std::vector<int64_t> vec_shape = {channels};
  const std::vector<int64_t> chan_shape = ChannelShape(op->nhwc, channels);
  const int64_t rest = ElementCount(x_shape) / (channels > 0 ? channels : 1);

  if (op->is_training && rest < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FusedBatchNorm in training mode needs more than one "
                 "element per channel to estimate a variance.");
    return;
  }

  // Outputs: y, batch_mean, batch_variance, reserve_space_1, reserve_space_2,
  // and on v3 a scalar reserve_space_3 that exists only for the cuDNN path.
  const int num_outputs = TF_NumOutputs(ctx);
  ScopedTensor y, out_mean, out_var, saved_mean, saved_var;
  y.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, x_shape.data(), 4,
      static_cast<size_t>(ElementCount(x_shape)) * TF_DataTypeSize(op->dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  const size_t vec_bytes =
      static_cast<size_t>(channels) * TF_DataTypeSize(op->param_dtype);
  out_mean.reset(TF_AllocateOutput(ctx, 1, op->param_dtype, vec_shape.data(), 1,
                                   vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  out_var.reset(TF_AllocateOutput(ctx, 2, op->param_dtype, vec_shape.data(), 1,
                                  vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  saved_mean.reset(TF_AllocateOutput(ctx, 3, op->param_dtype, vec_shape.data(),
                                     1, vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  saved_var.reset(TF_AllocateOutput(ctx, 4, op->param_dtype, vec_shape.data(),
                                    1, vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (num_outputs > 5 &&
      !AllocateEmptyScalar(ctx, 5, op->param_dtype, status)) {
    return;
  }
  if (ElementCount(x_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType x_type, u_type;
  if (!MPSTypeFor(op->dtype, &x_type, status)) return;
  if (!MPSTypeFor(op->param_dtype, &u_type, status)) return;

  std::string key = "FusedBatchNorm";
  AppendShapeToKey(x_shape, &key);
  key.append(op->nhwc ? "/NHWC" : "/NCHW");
  key.append(op->is_training ? "/train" : "/infer");
  key.append("/e").append(std::to_string(op->epsilon));
  key.append("/f").append(std::to_string(op->exponential_avg_factor));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/u").append(std::to_string(static_cast<int>(op->param_dtype)));
  key.append(op->has_side_input ? "/side" : "/plain");
  key.append(op->relu ? "/relu" : "/linear");

  const bool training = op->is_training;
  const bool nhwc = op->nhwc;
  const float epsilon = op->epsilon;
  const float factor = op->exponential_avg_factor;
  const bool has_side_input = op->has_side_input;
  const bool relu = op->relu;
  const double bessel =
      rest > 1 ? static_cast<double>(rest) / static_cast<double>(rest - 1) : 1.0;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* xt = [g placeholderWithShape:MPSShape(x_shape)
                                            dataType:x_type
                                                name:nil];
        MPSGraphTensor* scale_t = [g placeholderWithShape:MPSShape(chan_shape)
                                                 dataType:u_type
                                                     name:nil];
        MPSGraphTensor* offset_t = [g placeholderWithShape:MPSShape(chan_shape)
                                                  dataType:u_type
                                                      name:nil];
        MPSGraphTensor* mean_in = [g placeholderWithShape:MPSShape(chan_shape)
                                                 dataType:u_type
                                                     name:nil];
        MPSGraphTensor* var_in = [g placeholderWithShape:MPSShape(chan_shape)
                                                dataType:u_type
                                                    name:nil];
        [out->inputs addObject:xt];
        [out->inputs addObject:scale_t];
        [out->inputs addObject:offset_t];
        [out->inputs addObject:mean_in];
        [out->inputs addObject:var_in];

        // Statistics are accumulated in U, which for mixed precision is wider
        // than the data; casting first is what makes that meaningful.
        MPSGraphTensor* xu = x_type == u_type
                                 ? xt
                                 : [g castTensor:xt toType:u_type name:nil];

        MPSGraphTensor* mean;
        MPSGraphTensor* variance;
        if (training) {
          mean = [g meanOfTensor:xu axes:ReduceAxes(nhwc) name:nil];
          variance = [g varianceOfTensor:xu
                              meanTensor:mean
                                    axes:ReduceAxes(nhwc)
                                    name:nil];
        } else {
          mean = mean_in;
          variance = var_in;
        }

        MPSGraphTensor* yt = [g normalizationWithTensor:xu
                                            meanTensor:mean
                                        varianceTensor:variance
                                           gammaTensor:scale_t
                                            betaTensor:offset_t
                                               epsilon:epsilon
                                                  name:nil];
        if (x_type != u_type) yt = [g castTensor:yt toType:x_type name:nil];
        // The folded pair, in the order the fusion defines: the side input is
        // added to the normalised result, and the activation sees the sum.
        if (has_side_input) {
          MPSGraphTensor* side = [g placeholderWithShape:MPSShape(x_shape)
                                                dataType:x_type
                                                    name:nil];
          [out->inputs addObject:side];
          yt = [g additionWithPrimaryTensor:yt secondaryTensor:side name:nil];
        }
        if (relu) {
          yt = [g maximumWithPrimaryTensor:yt
                          secondaryTensor:[g constantWithScalar:0.0
                                                       dataType:x_type]
                                     name:nil];
        }

        // The reported batch statistics. In training these are the new
        // estimates, with Bessel's correction on the variance and an optional
        // blend with the incoming values; in inference they pass through.
        MPSGraphTensor* reported_mean = mean;
        MPSGraphTensor* reported_var = variance;
        if (training) {
          MPSGraphTensor* adjusted =
              [g multiplicationWithPrimaryTensor:variance
                                 secondaryTensor:[g constantWithScalar:bessel
                                                              dataType:u_type]
                                            name:nil];
          if (factor == 1.0f) {
            reported_var = adjusted;
          } else {
            MPSGraphTensor* f = [g constantWithScalar:factor dataType:u_type];
            MPSGraphTensor* one_minus =
                [g constantWithScalar:1.0 - factor dataType:u_type];
            reported_var = [g
                additionWithPrimaryTensor:
                    [g multiplicationWithPrimaryTensor:one_minus
                                       secondaryTensor:var_in
                                                  name:nil]
                          secondaryTensor:
                              [g multiplicationWithPrimaryTensor:f
                                                 secondaryTensor:adjusted
                                                            name:nil]
                                     name:nil];
            reported_mean = [g
                additionWithPrimaryTensor:
                    [g multiplicationWithPrimaryTensor:one_minus
                                       secondaryTensor:mean_in
                                                  name:nil]
                          secondaryTensor:
                              [g multiplicationWithPrimaryTensor:f
                                                 secondaryTensor:mean
                                                            name:nil]
                                     name:nil];
          }
        }

        // Core wants the per-channel outputs as flat vectors.
        NSArray<NSNumber*>* flat = MPSShape(vec_shape);
        [out->outputs addObject:yt];
        [out->outputs addObject:[g reshapeTensor:reported_mean
                                       withShape:flat
                                            name:nil]];
        [out->outputs addObject:[g reshapeTensor:reported_var
                                       withShape:flat
                                            name:nil]];
        [out->outputs addObject:[g reshapeTensor:mean withShape:flat name:nil]];
        [out->outputs
            addObject:[g reshapeTensor:variance withShape:flat name:nil]];
      },
      status);
  if (cached == nullptr) return;

  // The per-channel inputs are stored flat but fed with the broadcast shape.
  BufferSlice scale_slice, offset_slice, mean_slice, var_slice;
  if (!SliceForTensor(scale.get(), &scale_slice, status)) return;
  if (!SliceForTensor(offset.get(), &offset_slice, status)) return;
  if (!SliceForTensor(in_mean.get(), &mean_slice, status)) return;
  if (!SliceForTensor(in_variance.get(), &var_slice, status)) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(x.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* scale_data =
      TensorDataFor(scale_slice, chan_shape, op->param_dtype, device, status);
  if (scale_data == nil) return;
  MPSGraphTensorData* offset_data =
      TensorDataFor(offset_slice, chan_shape, op->param_dtype, device, status);
  if (offset_data == nil) return;
  MPSGraphTensorData* mean_data =
      TensorDataFor(mean_slice, chan_shape, op->param_dtype, device, status);
  if (mean_data == nil) return;
  MPSGraphTensorData* var_data =
      TensorDataFor(var_slice, chan_shape, op->param_dtype, device, status);
  if (var_data == nil) return;

  MPSGraphTensorData* y_data =
      TensorDataForTensor(y.get(), op->dtype, device, status);
  if (y_data == nil) return;
  MPSGraphTensorData* om_data =
      TensorDataForTensor(out_mean.get(), op->param_dtype, device, status);
  if (om_data == nil) return;
  MPSGraphTensorData* ov_data =
      TensorDataForTensor(out_var.get(), op->param_dtype, device, status);
  if (ov_data == nil) return;
  MPSGraphTensorData* sm_data =
      TensorDataForTensor(saved_mean.get(), op->param_dtype, device, status);
  if (sm_data == nil) return;
  MPSGraphTensorData* sv_data =
      TensorDataForTensor(saved_var.get(), op->param_dtype, device, status);
  if (sv_data == nil) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray
      arrayWithObjects:x_data, scale_data, offset_data, mean_data, var_data,
                       nil];
  if (op->has_side_input) {
    MPSGraphTensorData* side_data =
        TensorDataForTensor(side_input.get(), op->dtype, device, status);
    if (side_data == nil) return;
    [feeds addObject:side_data];
  }
  RunGraph(stream, *cached, feeds,
           @[ y_data, om_data, ov_data, sm_data, sv_data ], status);
}

/*** GRADIENT ***/

void FusedBatchNormGrad_ComputeImpl(BatchNormOp* op, TF_OpKernelContext* ctx,
                                    TF_Status* status) {
  ScopedTensor dy, x, scale, saved_mean, saved_var;
  TF_GetInput(ctx, 0, dy.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, x.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, scale.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, saved_mean.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 4, saved_var.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  // The Ex gradient is handed the activation's own output so it can undo it,
  // at the end of the input list rather than among the statistics.
  ScopedTensor activation_output;
  const bool undo_relu = op->relu && TF_NumInputs(ctx) > 7;
  if (undo_relu) {
    TF_GetInput(ctx, 7, activation_output.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  const std::vector<int64_t> x_shape = ShapeOf(x.get());
  if (x_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: FusedBatchNormGrad expects a rank-4 input.");
    return;
  }
  const int64_t channels = op->nhwc ? x_shape[3] : x_shape[1];
  const std::vector<int64_t> vec_shape = {channels};
  const std::vector<int64_t> chan_shape = ChannelShape(op->nhwc, channels);
  const int64_t rest = ElementCount(x_shape) / (channels > 0 ? channels : 1);

  ScopedTensor dx, dscale, doffset;
  dx.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, x_shape.data(), 4,
      static_cast<size_t>(ElementCount(x_shape)) * TF_DataTypeSize(op->dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  const size_t vec_bytes =
      static_cast<size_t>(channels) * TF_DataTypeSize(op->param_dtype);
  dscale.reset(TF_AllocateOutput(ctx, 1, op->param_dtype, vec_shape.data(), 1,
                                 vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  doffset.reset(TF_AllocateOutput(ctx, 2, op->param_dtype, vec_shape.data(), 1,
                                  vec_bytes, status));
  if (TF_GetCode(status) != TF_OK) return;
  // reserve_space_4 and _5 exist only for the cuDNN path and stay empty. The
  // Ex form adds a sixth output, the side input's gradient, which is a real
  // tensor rather than a placeholder.
  const int side_output =
      op->has_side_input && TF_NumOutputs(ctx) > 5 ? 5 : -1;
  ScopedTensor dside;
  for (int i = 3; i < TF_NumOutputs(ctx); ++i) {
    if (i == side_output) {
      dside.reset(TF_AllocateOutput(
          ctx, i, op->dtype, x_shape.data(), 4,
          static_cast<size_t>(ElementCount(x_shape)) *
              TF_DataTypeSize(op->dtype),
          status));
      if (TF_GetCode(status) != TF_OK) return;
      continue;
    }
    if (!AllocateEmptyScalar(ctx, i, op->param_dtype, status)) return;
  }
  if (ElementCount(x_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType x_type, u_type;
  if (!MPSTypeFor(op->dtype, &x_type, status)) return;
  if (!MPSTypeFor(op->param_dtype, &u_type, status)) return;

  std::string key = "FusedBatchNormGrad";
  AppendShapeToKey(x_shape, &key);
  key.append(op->nhwc ? "/NHWC" : "/NCHW");
  key.append(op->is_training ? "/train" : "/infer");
  key.append("/e").append(std::to_string(op->epsilon));
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/u").append(std::to_string(static_cast<int>(op->param_dtype)));
  key.append(undo_relu ? "/relu" : "/linear");
  key.append(op->has_side_input && TF_NumOutputs(ctx) > 5 ? "/side" : "/plain");

  const bool nhwc = op->nhwc;
  const bool training = op->is_training;
  const float epsilon = op->epsilon;
  const double inv_rest = rest > 0 ? 1.0 / static_cast<double>(rest) : 0.0;
  const bool wants_side = side_output >= 0;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        NSArray<NSNumber*>* axes = ReduceAxes(nhwc);

        MPSGraphTensor* dy_t = [g placeholderWithShape:MPSShape(x_shape)
                                              dataType:x_type
                                                  name:nil];
        MPSGraphTensor* x_t = [g placeholderWithShape:MPSShape(x_shape)
                                             dataType:x_type
                                                 name:nil];
        MPSGraphTensor* scale_t = [g placeholderWithShape:MPSShape(chan_shape)
                                                 dataType:u_type
                                                     name:nil];
        MPSGraphTensor* mean_t = [g placeholderWithShape:MPSShape(chan_shape)
                                                dataType:u_type
                                                    name:nil];
        MPSGraphTensor* var_t = [g placeholderWithShape:MPSShape(chan_shape)
                                               dataType:u_type
                                                   name:nil];
        [out->inputs addObject:dy_t];
        [out->inputs addObject:x_t];
        [out->inputs addObject:scale_t];
        [out->inputs addObject:mean_t];
        [out->inputs addObject:var_t];

        // The rectifier's gradient is read off its own output: where the
        // output is positive the input passed through, and elsewhere it did
        // not. That is why the Ex gradient is given y at all.
        MPSGraphTensor* dy_pre = dy_t;
        if (undo_relu) {
          MPSGraphTensor* y_t = [g placeholderWithShape:MPSShape(x_shape)
                                               dataType:x_type
                                                   name:nil];
          [out->inputs addObject:y_t];
          dy_pre = [g
              selectWithPredicateTensor:[g greaterThanWithPrimaryTensor:y_t
                                                       secondaryTensor:
                                                           [g constantWithScalar:0.0
                                                                        dataType:x_type]
                                                                  name:nil]
                    truePredicateTensor:dy_t
                   falsePredicateTensor:[g constantWithScalar:0.0
                                                     dataType:x_type]
                                   name:nil];
        }

        MPSGraphTensor* dyu = x_type == u_type
                                  ? dy_pre
                                  : [g castTensor:dy_pre toType:u_type name:nil];
        MPSGraphTensor* xu = x_type == u_type
                                 ? x_t
                                 : [g castTensor:x_t toType:u_type name:nil];

        MPSGraphTensor* eps = [g constantWithScalar:epsilon dataType:u_type];
        MPSGraphTensor* inv_std = [g
            reciprocalSquareRootWithTensor:[g additionWithPrimaryTensor:var_t
                                                       secondaryTensor:eps
                                                                  name:nil]
                                      name:nil];
        MPSGraphTensor* x_center =
            [g subtractionWithPrimaryTensor:xu secondaryTensor:mean_t name:nil];

        // dscale = sum(dy * x_center * inv_std), doffset = sum(dy)
        MPSGraphTensor* dy_xc =
            [g multiplicationWithPrimaryTensor:dyu
                               secondaryTensor:x_center
                                          name:nil];
        MPSGraphTensor* sum_dy_xc = [g reductionSumWithTensor:dy_xc
                                                         axes:axes
                                                         name:nil];
        MPSGraphTensor* dscale_t =
            [g multiplicationWithPrimaryTensor:sum_dy_xc
                               secondaryTensor:inv_std
                                          name:nil];
        MPSGraphTensor* sum_dy = [g reductionSumWithTensor:dyu
                                                      axes:axes
                                                      name:nil];

        MPSGraphTensor* dx_u;
        if (training) {
          MPSGraphTensor* inv_n =
              [g constantWithScalar:inv_rest dataType:u_type];
          MPSGraphTensor* dy_mean =
              [g multiplicationWithPrimaryTensor:sum_dy
                                 secondaryTensor:inv_n
                                            name:nil];
          MPSGraphTensor* dy_centered =
              [g subtractionWithPrimaryTensor:dyu
                              secondaryTensor:dy_mean
                                         name:nil];
          MPSGraphTensor* dy_xc_mean =
              [g multiplicationWithPrimaryTensor:sum_dy_xc
                                 secondaryTensor:inv_n
                                            name:nil];
          MPSGraphTensor* coef2 =
              [g multiplicationWithPrimaryTensor:[g squareWithTensor:inv_std
                                                                name:nil]
                                 secondaryTensor:dy_xc_mean
                                            name:nil];
          MPSGraphTensor* inner =
              [g subtractionWithPrimaryTensor:dy_centered
                              secondaryTensor:
                                  [g multiplicationWithPrimaryTensor:x_center
                                                     secondaryTensor:coef2
                                                                name:nil]
                                         name:nil];
          MPSGraphTensor* coef1 =
              [g multiplicationWithPrimaryTensor:scale_t
                                 secondaryTensor:inv_std
                                            name:nil];
          dx_u = [g multiplicationWithPrimaryTensor:coef1
                                    secondaryTensor:inner
                                               name:nil];
        } else {
          // With frozen statistics the batch terms vanish and only the
          // per-element scaling survives.
          MPSGraphTensor* coef1 =
              [g multiplicationWithPrimaryTensor:scale_t
                                 secondaryTensor:inv_std
                                            name:nil];
          dx_u = [g multiplicationWithPrimaryTensor:dyu
                                    secondaryTensor:coef1
                                               name:nil];
        }

        MPSGraphTensor* dx_t =
            x_type == u_type ? dx_u : [g castTensor:dx_u toType:x_type name:nil];
        NSArray<NSNumber*>* flat = MPSShape(vec_shape);
        [out->outputs addObject:dx_t];
        [out->outputs
            addObject:[g reshapeTensor:dscale_t withShape:flat name:nil]];
        [out->outputs
            addObject:[g reshapeTensor:sum_dy withShape:flat name:nil]];
        // A side input is added after the normalisation, so its gradient is
        // whatever reached that addition: the gradient with the activation
        // already undone, unchanged.
        if (wants_side) [out->outputs addObject:dy_pre];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice scale_slice, mean_slice, var_slice;
  if (!SliceForTensor(scale.get(), &scale_slice, status)) return;
  if (!SliceForTensor(saved_mean.get(), &mean_slice, status)) return;
  if (!SliceForTensor(saved_var.get(), &var_slice, status)) return;

  MPSGraphTensorData* dy_data =
      TensorDataForTensor(dy.get(), op->dtype, device, status);
  if (dy_data == nil) return;
  MPSGraphTensorData* x_data =
      TensorDataForTensor(x.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* scale_data =
      TensorDataFor(scale_slice, chan_shape, op->param_dtype, device, status);
  if (scale_data == nil) return;
  MPSGraphTensorData* mean_data =
      TensorDataFor(mean_slice, chan_shape, op->param_dtype, device, status);
  if (mean_data == nil) return;
  MPSGraphTensorData* var_data =
      TensorDataFor(var_slice, chan_shape, op->param_dtype, device, status);
  if (var_data == nil) return;

  MPSGraphTensorData* dx_data =
      TensorDataForTensor(dx.get(), op->dtype, device, status);
  if (dx_data == nil) return;
  MPSGraphTensorData* dscale_data =
      TensorDataForTensor(dscale.get(), op->param_dtype, device, status);
  if (dscale_data == nil) return;
  MPSGraphTensorData* doffset_data =
      TensorDataForTensor(doffset.get(), op->param_dtype, device, status);
  if (doffset_data == nil) return;

  NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray
      arrayWithObjects:dy_data, x_data, scale_data, mean_data, var_data, nil];
  if (undo_relu) {
    MPSGraphTensorData* y_data =
        TensorDataForTensor(activation_output.get(), op->dtype, device,
                            status);
    if (y_data == nil) return;
    [feeds addObject:y_data];
  }
  NSMutableArray<MPSGraphTensorData*>* results = [NSMutableArray
      arrayWithObjects:dx_data, dscale_data, doffset_data, nil];
  if (side_output >= 0) {
    MPSGraphTensorData* dside_data =
        TensorDataForTensor(dside.get(), op->dtype, device, status);
    if (dside_data == nil) return;
    [results addObject:dside_data];
  }
  RunGraph(stream, *cached, feeds, results, status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<BatchNormOp*>(kernel);                             \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(FusedBatchNorm_Compute, FusedBatchNorm_ComputeImpl)
METAL_COMPUTE(FusedBatchNormGrad_Compute, FusedBatchNormGrad_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name, void (*compute)(void*, TF_OpKernelContext*),
              TF_DataType dtype, bool has_u, TF_DataType u_dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &BatchNormOp_Create,
                          compute, &BatchNormOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && has_u) {
    TF_KernelBuilder_TypeConstraint(builder, "U", u_dtype, status);
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

void RegisterMetalBatchNormKernels() {
  // v1 has no U attribute; v2 and v3 keep the statistics in float32 even when
  // the data is float16, which is what mixed-precision training expects.
  Register("FusedBatchNorm", &FusedBatchNorm_Compute, TF_FLOAT, false, TF_FLOAT,
           "MetalFusedBatchNormFloat");
  Register("FusedBatchNormGrad", &FusedBatchNormGrad_Compute, TF_FLOAT, false,
           TF_FLOAT, "MetalFusedBatchNormGradFloat");

  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    Register("FusedBatchNormV2", &FusedBatchNorm_Compute, kDTypes[i], true,
             TF_FLOAT, std::string("MetalFusedBatchNormV2") + kSuffixes[i]);
    Register("FusedBatchNormV3", &FusedBatchNorm_Compute, kDTypes[i], true,
             TF_FLOAT, std::string("MetalFusedBatchNormV3") + kSuffixes[i]);
    Register("FusedBatchNormGradV2", &FusedBatchNormGrad_Compute, kDTypes[i],
             true, TF_FLOAT,
             std::string("MetalFusedBatchNormGradV2") + kSuffixes[i]);
    // The Ex form is the same pass with an optional side input and an
    // optional activation folded in; the kernel reads both from its
    // attributes, so it needs no separate compute.
    Register("_FusedBatchNormEx", &FusedBatchNorm_Compute, kDTypes[i], true,
             TF_FLOAT, std::string("Metal_FusedBatchNormEx") + kSuffixes[i]);
    Register("FusedBatchNormGradV3", &FusedBatchNormGrad_Compute, kDTypes[i],
             true, TF_FLOAT,
             std::string("MetalFusedBatchNormGradV3") + kSuffixes[i]);
    Register("_FusedBatchNormGradEx", &FusedBatchNormGrad_Compute, kDTypes[i],
             true, TF_FLOAT,
             std::string("Metal_FusedBatchNormGradEx") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
