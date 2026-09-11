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

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  return count;
}

// Allocates output `index` with `shape`, returning false with `status` set.
bool AllocateOutput(TF_OpKernelContext* ctx, int index,
                    const std::vector<int64_t>& shape, TF_DataType dtype,
                    ScopedTensor* out, TF_Status* status) {
  out->reset(TF_AllocateOutput(
      ctx, index, dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(ElementCount(shape)) * TF_DataTypeSize(dtype),
      status));
  return TF_GetCode(status) == TF_OK;
}

// State shared by the kernels here that need only the element dtype.
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

/*** RELU ***/

void Relu_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor features;
  TF_GetInput(ctx, 0, features.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(features.get());
  ScopedTensor output;
  if (!AllocateOutput(ctx, 0, shape, op->dtype, &output, status)) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Relu";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* source = [out->graph placeholderWithShape:MPSShape(shape)
                                                         dataType:mps_dtype
                                                             name:nil];
        [out->inputs addObject:source];
        [out->outputs addObject:[out->graph reLUWithTensor:source name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(features.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

void ReluGrad_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor gradients;
  ScopedTensor features;
  TF_GetInput(ctx, 0, gradients.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, features.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(features.get());
  ScopedTensor output;
  if (!AllocateOutput(ctx, 0, shape, op->dtype, &output, status)) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "ReluGrad";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* gradient =
            [out->graph placeholderWithShape:MPSShape(shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* source = [out->graph placeholderWithShape:MPSShape(shape)
                                                         dataType:mps_dtype
                                                             name:nil];
        [out->inputs addObject:gradient];
        [out->inputs addObject:source];
        [out->outputs addObject:[out->graph reLUGradientWithIncomingGradient:gradient
                                                                sourceTensor:source
                                                                        name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* grad_data =
      TensorDataForTensor(gradients.get(), op->dtype, device, status);
  if (grad_data == nil) return;
  MPSGraphTensorData* feat_data =
      TensorDataForTensor(features.get(), op->dtype, device, status);
  if (feat_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ grad_data, feat_data ], @[ out_data ], status);
}

/*** BIAS ADD ***/

struct BiasOp {
  TF_DataType dtype = TF_FLOAT;
  bool nhwc = true;
};

void* BiasOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BiasOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  char format[8] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format,
                                        sizeof(format) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    // data_format is optional on BiasAdd and defaults to NHWC.
    TF_SetStatus(status, TF_OK, "");
    op->nhwc = true;
  } else {
    op->nhwc = std::strcmp(format, "NCHW") != 0;
  }
  TF_DeleteStatus(status);
  return op;
}

void BiasOp_Delete(void* kernel) { delete static_cast<BiasOp*>(kernel); }

// Shape the bias vector has to take so that MPSGraph broadcasts it over the
// right axis: length C on the channel axis, 1 everywhere else.
std::vector<int64_t> BiasBroadcastShape(const std::vector<int64_t>& value_shape,
                                        bool nhwc, int64_t channels) {
  std::vector<int64_t> shape(value_shape.size(), 1);
  shape[nhwc ? value_shape.size() - 1 : 1] = channels;
  return shape;
}

void BiasAdd_ComputeImpl(BiasOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor value;
  ScopedTensor bias;
  TF_GetInput(ctx, 0, value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, bias.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(value.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BiasAdd expects a value of rank at least 2.");
    return;
  }
  const int64_t channels = TF_TensorElementCount(bias.get());
  const std::vector<int64_t> bias_shape =
      BiasBroadcastShape(shape, op->nhwc, channels);

  ScopedTensor output;
  if (!AllocateOutput(ctx, 0, shape, op->dtype, &output, status)) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "BiasAdd";
  AppendShapeToKey(shape, &key);
  AppendShapeToKey(bias_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* value_tensor =
            [out->graph placeholderWithShape:MPSShape(shape)
                                    dataType:mps_dtype
                                        name:nil];
        // Fed as the broadcast shape rather than as a flat vector, so MPSGraph
        // lines the bias up with the channel axis under either data format.
        MPSGraphTensor* bias_tensor =
            [out->graph placeholderWithShape:MPSShape(bias_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:value_tensor];
        [out->inputs addObject:bias_tensor];
        [out->outputs
            addObject:[out->graph additionWithPrimaryTensor:value_tensor
                                           secondaryTensor:bias_tensor
                                                      name:nil]];
      },
      status);
  if (cached == nullptr) return;

  BufferSlice bias_slice;
  if (!SliceForTensor(bias.get(), &bias_slice, status)) return;

  MPSGraphTensorData* value_data =
      TensorDataForTensor(value.get(), op->dtype, device, status);
  if (value_data == nil) return;
  MPSGraphTensorData* bias_data =
      TensorDataFor(bias_slice, bias_shape, op->dtype, device, status);
  if (bias_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ value_data, bias_data ], @[ out_data ], status);
}

void BiasAddGrad_ComputeImpl(BiasOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor out_backprop;
  TF_GetInput(ctx, 0, out_backprop.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(out_backprop.get());
  if (shape.size() < 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: BiasAddGrad expects a rank of at least 2.");
    return;
  }
  const size_t channel_axis = op->nhwc ? shape.size() - 1 : 1;
  const std::vector<int64_t> bias_shape = {shape[channel_axis]};

  ScopedTensor output;
  if (!AllocateOutput(ctx, 0, bias_shape, op->dtype, &output, status)) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "BiasAddGrad";
  AppendShapeToKey(shape, &key);
  key.append(op->nhwc ? "/NHWC" : "/NCHW");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  // Sum over every axis except the channel one, then drop the size-1 axes the
  // reduction leaves behind so the result matches the bias vector's shape.
  NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != channel_axis) [axes addObject:@(static_cast<NSInteger>(i))];
  }
  const std::vector<int64_t> flat_shape = bias_shape;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* gradient =
            [out->graph placeholderWithShape:MPSShape(shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* summed = [out->graph reductionSumWithTensor:gradient
                                                               axes:axes
                                                               name:nil];
        MPSGraphTensor* flat = [out->graph reshapeTensor:summed
                                               withShape:MPSShape(flat_shape)
                                                    name:nil];
        [out->inputs addObject:gradient];
        [out->outputs addObject:flat];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* grad_data =
      TensorDataForTensor(out_backprop.get(), op->dtype, device, status);
  if (grad_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ grad_data ], @[ out_data ], status);
}

/*** SOFTMAX ***/

void Softmax_ComputeImpl(DTypeOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor logits;
  TF_GetInput(ctx, 0, logits.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(logits.get());
  if (shape.size() < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Softmax expects a rank of at least 1.");
    return;
  }
  ScopedTensor output;
  if (!AllocateOutput(ctx, 0, shape, op->dtype, &output, status)) return;
  if (ElementCount(shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Softmax";
  AppendShapeToKey(shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  const NSInteger axis = static_cast<NSInteger>(shape.size()) - 1;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* source = [out->graph placeholderWithShape:MPSShape(shape)
                                                         dataType:mps_dtype
                                                             name:nil];
        [out->inputs addObject:source];
        [out->outputs addObject:[out->graph softMaxWithTensor:source
                                                          axis:axis
                                                          name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(logits.get(), op->dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

/*** SOFTMAX CROSS ENTROPY ***/

// Builds a numerically stable log-softmax over the last axis.
//
// Written out rather than taking the logarithm of MPSGraph's softMax: a
// probability that underflows to zero would give -inf, and the whole point of
// fusing the softmax with the cross entropy is to avoid exactly that. This is
// the same max-subtraction TensorFlow's own kernel uses.
MPSGraphTensor* LogSoftMax(MPSGraph* graph, MPSGraphTensor* logits,
                           NSInteger axis) {
  MPSGraphTensor* max = [graph reductionMaximumWithTensor:logits
                                                     axis:axis
                                                     name:nil];
  MPSGraphTensor* shifted = [graph subtractionWithPrimaryTensor:logits
                                               secondaryTensor:max
                                                          name:nil];
  MPSGraphTensor* exponent = [graph exponentWithTensor:shifted name:nil];
  MPSGraphTensor* sum = [graph reductionSumWithTensor:exponent
                                                 axis:axis
                                                 name:nil];
  MPSGraphTensor* log_sum = [graph logarithmWithTensor:sum name:nil];
  return [graph subtractionWithPrimaryTensor:shifted
                             secondaryTensor:log_sum
                                        name:nil];
}

struct XentOp {
  TF_DataType dtype = TF_FLOAT;
  TF_DataType label_dtype = TF_INT32;
  bool sparse = false;
};

template <bool kSparse>
void* XentOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new XentOp();
  op->sparse = kSparse;
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) == TF_OK && kSparse) {
    TF_OpKernelConstruction_GetAttrType(ctx, "Tlabels", &op->label_dtype,
                                        status);
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

void XentOp_Delete(void* kernel) { delete static_cast<XentOp*>(kernel); }

void Xent_ComputeImpl(XentOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor features;
  ScopedTensor labels;
  TF_GetInput(ctx, 0, features.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, labels.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> logits_shape = ShapeOf(features.get());
  if (logits_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: softmax cross entropy expects rank-2 logits.");
    return;
  }
  const std::vector<int64_t> label_shape = ShapeOf(labels.get());
  const int64_t batch = logits_shape[0];
  const int64_t classes = logits_shape[1];

  // Two outputs: the per-example loss and the gradient with respect to the
  // logits, which TensorFlow expects this op to produce together.
  const std::vector<int64_t> loss_shape = {batch};
  ScopedTensor loss;
  ScopedTensor backprop;
  if (!AllocateOutput(ctx, 0, loss_shape, op->dtype, &loss, status)) return;
  if (!AllocateOutput(ctx, 1, logits_shape, op->dtype, &backprop, status)) {
    return;
  }
  if (batch == 0 || classes == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;
  MPSDataType label_mps_dtype = mps_dtype;
  if (op->sparse && !MPSTypeFor(op->label_dtype, &label_mps_dtype, status)) {
    return;
  }

  std::string key = op->sparse ? "SparseXent" : "Xent";
  AppendShapeToKey(logits_shape, &key);
  AppendShapeToKey(label_shape, &key);
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));
  key.append("/l").append(std::to_string(static_cast<int>(op->label_dtype)));

  const bool sparse = op->sparse;
  const NSUInteger depth = static_cast<NSUInteger>(classes);

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* logits =
            [out->graph placeholderWithShape:MPSShape(logits_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* label_input =
            [out->graph placeholderWithShape:MPSShape(label_shape)
                                    dataType:label_mps_dtype
                                        name:nil];

        MPSGraphTensor* one_hot = label_input;
        if (sparse) {
          // Sparse labels are class indices; the dense formulation below wants
          // a probability distribution per row.
          MPSGraphTensor* indices =
              [out->graph castTensor:label_input
                              toType:MPSDataTypeInt32
                                name:nil];
          one_hot = [out->graph oneHotWithIndicesTensor:indices
                                                  depth:depth
                                                   axis:1
                                               dataType:mps_dtype
                                                onValue:1.0
                                               offValue:0.0
                                                   name:nil];
        }

        MPSGraphTensor* log_softmax = LogSoftMax(out->graph, logits, 1);
        MPSGraphTensor* weighted =
            [out->graph multiplicationWithPrimaryTensor:one_hot
                                        secondaryTensor:log_softmax
                                                   name:nil];
        MPSGraphTensor* summed = [out->graph reductionSumWithTensor:weighted
                                                               axis:1
                                                               name:nil];
        MPSGraphTensor* negated = [out->graph negativeWithTensor:summed
                                                            name:nil];
        // The reduction keeps a size-1 axis; TensorFlow wants a flat [batch].
        MPSGraphTensor* loss_tensor =
            [out->graph reshapeTensor:negated
                            withShape:MPSShape(loss_shape)
                                 name:nil];

        // softmax(logits) - labels, recovered from the log-softmax already
        // computed rather than by running the softmax a second time.
        MPSGraphTensor* softmax = [out->graph exponentWithTensor:log_softmax
                                                            name:nil];
        MPSGraphTensor* grad =
            [out->graph subtractionWithPrimaryTensor:softmax
                                     secondaryTensor:one_hot
                                                name:nil];

        [out->inputs addObject:logits];
        [out->inputs addObject:label_input];
        [out->outputs addObject:loss_tensor];
        [out->outputs addObject:grad];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* logits_data =
      TensorDataForTensor(features.get(), op->dtype, device, status);
  if (logits_data == nil) return;
  MPSGraphTensorData* labels_data = TensorDataForTensor(
      labels.get(), op->sparse ? op->label_dtype : op->dtype, device, status);
  if (labels_data == nil) return;
  MPSGraphTensorData* loss_data =
      TensorDataForTensor(loss.get(), op->dtype, device, status);
  if (loss_data == nil) return;
  MPSGraphTensorData* grad_data =
      TensorDataForTensor(backprop.get(), op->dtype, device, status);
  if (grad_data == nil) return;

  RunGraph(stream, *cached, @[ logits_data, labels_data ],
           @[ loss_data, grad_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_DEFINE_COMPUTE(NAME, STATE, IMPL)                          \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                     \
    ScopedAutoreleasePool pool;                                          \
    TF_Status* status = TF_NewStatus();                                  \
    STATE* state = static_cast<STATE*>(kernel);                          \
    if (state == nullptr) {                                              \
      TF_SetStatus(status, TF_INTERNAL,                                  \
                   "Metal: kernel has no state; construction failed.");  \
    } else {                                                             \
      IMPL(state, ctx, status);                                          \
    }                                                                    \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                             \
  }

METAL_DEFINE_COMPUTE(Relu_Compute, DTypeOp, Relu_ComputeImpl)
METAL_DEFINE_COMPUTE(ReluGrad_Compute, DTypeOp, ReluGrad_ComputeImpl)
METAL_DEFINE_COMPUTE(Softmax_Compute, DTypeOp, Softmax_ComputeImpl)
METAL_DEFINE_COMPUTE(BiasAdd_Compute, BiasOp, BiasAdd_ComputeImpl)
METAL_DEFINE_COMPUTE(BiasAddGrad_Compute, BiasOp, BiasAddGrad_ComputeImpl)
METAL_DEFINE_COMPUTE(Xent_Compute, XentOp, Xent_ComputeImpl)

#undef METAL_DEFINE_COMPUTE

void RegisterSimple(const char* op_name,
                    void* (*create)(TF_OpKernelConstruction*),
                    void (*compute)(void*, TF_OpKernelContext*),
                    void (*destroy)(void*), TF_DataType dtype,
                    const std::string& kernel_name,
                    const char* second_constraint_name = nullptr,
                    TF_DataType second_constraint = TF_FLOAT) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && second_constraint_name != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, second_constraint_name,
                                    second_constraint, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(kernel_name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << kernel_name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalNnKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};

  for (int i = 0; i < 2; ++i) {
    const TF_DataType dtype = kDTypes[i];
    const std::string suffix = kSuffixes[i];

    RegisterSimple("Relu", &DTypeOp_Create, &Relu_Compute, &DTypeOp_Delete,
                   dtype, "MetalRelu" + suffix);
    RegisterSimple("ReluGrad", &DTypeOp_Create, &ReluGrad_Compute,
                   &DTypeOp_Delete, dtype, "MetalReluGrad" + suffix);
    RegisterSimple("Softmax", &DTypeOp_Create, &Softmax_Compute,
                   &DTypeOp_Delete, dtype, "MetalSoftmax" + suffix);
    RegisterSimple("BiasAdd", &BiasOp_Create, &BiasAdd_Compute, &BiasOp_Delete,
                   dtype, "MetalBiasAdd" + suffix);
    RegisterSimple("BiasAddGrad", &BiasOp_Create, &BiasAddGrad_Compute,
                   &BiasOp_Delete, dtype, "MetalBiasAddGrad" + suffix);
    RegisterSimple("SoftmaxCrossEntropyWithLogits", &XentOp_Create<false>,
                   &Xent_Compute, &XentOp_Delete, dtype,
                   "MetalXent" + suffix);

    // Sparse labels arrive as either int32 or int64 depending on how the model
    // was written, and both are common.
    RegisterSimple("SparseSoftmaxCrossEntropyWithLogits",
                   &XentOp_Create<true>, &Xent_Compute, &XentOp_Delete, dtype,
                   "MetalSparseXent" + suffix + "Int32", "Tlabels", TF_INT32);
    RegisterSimple("SparseSoftmaxCrossEntropyWithLogits",
                   &XentOp_Create<true>, &Xent_Compute, &XentOp_Delete, dtype,
                   "MetalSparseXent" + suffix + "Int64", "Tlabels", TF_INT64);
  }
}

}  // namespace metal
}  // namespace tensorflow
