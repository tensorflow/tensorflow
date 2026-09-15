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

// Conv2D and its two gradients, on MPSGraph.
//
// The mapping is unusually direct: MPSGraph's padding styles are literally
// named MPSGraphPaddingStyleTF_SAME and TF_VALID and follow TensorFlow's
// output-size and padding-split rules, and its NHWC and HWIO layouts are
// TensorFlow's defaults for the input and the filter. So nothing is transposed
// or repacked here; the tensors are handed to MPS in the layout TensorFlow
// already stores them in.

MPSGraphConvolution2DOpDescriptor* DescriptorFor(const SpatialParams& params) {
  return [MPSGraphConvolution2DOpDescriptor
      descriptorWithStrideInX:static_cast<NSUInteger>(params.stride_w)
                    strideInY:static_cast<NSUInteger>(params.stride_h)
              dilationRateInX:static_cast<NSUInteger>(params.dilation_w)
              dilationRateInY:static_cast<NSUInteger>(params.dilation_h)
                       groups:1
                 paddingStyle:params.same_padding ? MPSGraphPaddingStyleTF_SAME
                                                  : MPSGraphPaddingStyleTF_VALID
                   dataLayout:params.nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                          : MPSGraphTensorNamedDataLayoutNCHW
                weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
}

void AppendParamsToKey(const SpatialParams& params, std::string* key) {
  key->append("/s").append(std::to_string(params.stride_h));
  key->push_back('x');
  key->append(std::to_string(params.stride_w));
  key->append("/d").append(std::to_string(params.dilation_h));
  key->push_back('x');
  key->append(std::to_string(params.dilation_w));
  key->append(params.same_padding ? "/SAME" : "/VALID");
  key->append(params.nhwc ? "/NHWC" : "/NCHW");
  key->append("/t").append(std::to_string(static_cast<int>(params.dtype)));
}

// Output extent along one spatial axis, following TensorFlow's rules. MPSGraph
// computes the same thing internally, but core needs the output tensor
// allocated before the graph runs.
int64_t ConvOutputExtent(int64_t input, int64_t filter, int stride,
                         int dilation, bool same_padding) {
  if (same_padding) {
    return (input + stride - 1) / stride;
  }
  const int64_t effective_filter = (filter - 1) * dilation + 1;
  if (input < effective_filter) return 0;
  return (input - effective_filter) / stride + 1;
}

// Reads a 4-element int32 shape from a host-memory tensor, which is how
// TensorFlow passes the target shape to the gradient ops.
bool ShapeFromHostTensor(TF_Tensor* tensor, std::vector<int64_t>* out,
                         const char* what, TF_Status* status) {
  if (TF_TensorElementCount(tensor) != 4 ||
      TF_TensorType(tensor) != TF_INT32) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 (std::string("Metal: ") + what +
                  " must be a 4-element int32 tensor.")
                     .c_str());
    return false;
  }
  const int32_t* values = static_cast<const int32_t*>(TF_TensorData(tensor));
  if (values == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 (std::string("Metal: ") + what + " has no data.").c_str());
    return false;
  }
  out->assign(values, values + 4);
  return true;
}

struct ConvOp {
  SpatialParams params;
};

template <bool kWantDilations>
void* ConvOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ConvOp();
  if (!ReadSpatialParams(ctx, kWantDilations, &op->params, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void ConvOp_Delete(void* kernel) { delete static_cast<ConvOp*>(kernel); }

/*** FORWARD ***/

void Conv2D_ComputeImpl(ConvOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Conv2D kernel has no state.");
    return;
  }

  ScopedTensor input;
  ScopedTensor filter;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  if (TF_NumDims(input.get()) != 4 || TF_NumDims(filter.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Conv2D expects a rank-4 input and filter.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> filter_shape = ShapeOf(filter.get());
  const int h_index = SpatialHeightIndex(op->params.nhwc);
  const int w_index = SpatialWidthIndex(op->params.nhwc);

  // Filter is HWIO: [height, width, in_channels, out_channels].
  std::vector<int64_t> out_shape = in_shape;
  out_shape[h_index] =
      ConvOutputExtent(in_shape[h_index], filter_shape[0], op->params.stride_h,
                       op->params.dilation_h, op->params.same_padding);
  out_shape[w_index] =
      ConvOutputExtent(in_shape[w_index], filter_shape[1], op->params.stride_w,
                       op->params.dilation_w, op->params.same_padding);
  out_shape[op->params.nhwc ? 3 : 1] = filter_shape[3];

  int64_t element_count = 1;
  for (int64_t dim : out_shape) element_count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->params.dtype, out_shape.data(), 4,
      static_cast<size_t>(element_count) * TF_DataTypeSize(op->params.dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (element_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "Conv2D";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(filter_shape, &key);
  AppendParamsToKey(op->params, &key);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;
  const SpatialParams params = op->params;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* source =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* weights =
            [out->graph placeholderWithShape:MPSShape(filter_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* result =
            [out->graph convolution2DWithSourceTensor:source
                                        weightsTensor:weights
                                           descriptor:DescriptorFor(params)
                                                 name:nil];
        [out->inputs addObject:source];
        [out->inputs addObject:weights];
        [out->outputs addObject:result];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* input_data =
      TensorDataForTensor(input.get(), op->params.dtype, device, status);
  if (input_data == nil) return;
  MPSGraphTensorData* filter_data =
      TensorDataForTensor(filter.get(), op->params.dtype, device, status);
  if (filter_data == nil) return;
  MPSGraphTensorData* output_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (output_data == nil) return;

  RunGraph(stream, *cached, @[ input_data, filter_data ], @[ output_data ],
           status);
}

/*** GRADIENT WITH RESPECT TO THE INPUT ***/

void Conv2DBackpropInput_ComputeImpl(ConvOp* op, TF_OpKernelContext* ctx,
                                     TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: Conv2DBackpropInput kernel has no state.");
    return;
  }

  ScopedTensor input_sizes;
  ScopedTensor filter;
  ScopedTensor out_backprop;
  TF_GetInput(ctx, 0, input_sizes.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, out_backprop.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> in_shape;
  if (!ShapeFromHostTensor(input_sizes.get(), &in_shape, "input_sizes",
                           status)) {
    return;
  }

  const std::vector<int64_t> filter_shape = ShapeOf(filter.get());
  const std::vector<int64_t> grad_shape = ShapeOf(out_backprop.get());

  int64_t element_count = 1;
  for (int64_t dim : in_shape) element_count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->params.dtype, in_shape.data(), 4,
      static_cast<size_t>(element_count) * TF_DataTypeSize(op->params.dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (element_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "Conv2DBackpropInput";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(filter_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendParamsToKey(op->params, &key);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;
  const SpatialParams params = op->params;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* gradient =
            [out->graph placeholderWithShape:MPSShape(grad_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* weights =
            [out->graph placeholderWithShape:MPSShape(filter_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* result = [out->graph
            convolution2DDataGradientWithIncomingGradientTensor:gradient
                                                  weightsTensor:weights
                                                    outputShape:MPSShape(
                                                                    in_shape)
                                   forwardConvolutionDescriptor:
                                       DescriptorFor(params)
                                                           name:nil];
        [out->inputs addObject:gradient];
        [out->inputs addObject:weights];
        [out->outputs addObject:result];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* gradient_data =
      TensorDataForTensor(out_backprop.get(), op->params.dtype, device, status);
  if (gradient_data == nil) return;
  MPSGraphTensorData* filter_data =
      TensorDataForTensor(filter.get(), op->params.dtype, device, status);
  if (filter_data == nil) return;
  MPSGraphTensorData* output_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (output_data == nil) return;

  RunGraph(stream, *cached, @[ gradient_data, filter_data ], @[ output_data ],
           status);
}

/*** GRADIENT WITH RESPECT TO THE FILTER ***/

void Conv2DBackpropFilter_ComputeImpl(ConvOp* op, TF_OpKernelContext* ctx,
                                      TF_Status* status) {
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: Conv2DBackpropFilter kernel has no state.");
    return;
  }

  ScopedTensor input;
  ScopedTensor filter_sizes;
  ScopedTensor out_backprop;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter_sizes.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, out_backprop.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> filter_shape;
  if (!ShapeFromHostTensor(filter_sizes.get(), &filter_shape, "filter_sizes",
                           status)) {
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> grad_shape = ShapeOf(out_backprop.get());

  int64_t element_count = 1;
  for (int64_t dim : filter_shape) element_count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->params.dtype, filter_shape.data(), 4,
      static_cast<size_t>(element_count) * TF_DataTypeSize(op->params.dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (element_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  std::string key = "Conv2DBackpropFilter";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(filter_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendParamsToKey(op->params, &key);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;
  const SpatialParams params = op->params;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* gradient =
            [out->graph placeholderWithShape:MPSShape(grad_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* source =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* result = [out->graph
            convolution2DWeightsGradientWithIncomingGradientTensor:gradient
                                                      sourceTensor:source
                                                       outputShape:
                                                           MPSShape(
                                                               filter_shape)
                                      forwardConvolutionDescriptor:
                                          DescriptorFor(params)
                                                              name:nil];
        [out->inputs addObject:gradient];
        [out->inputs addObject:source];
        [out->outputs addObject:result];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* gradient_data =
      TensorDataForTensor(out_backprop.get(), op->params.dtype, device, status);
  if (gradient_data == nil) return;
  MPSGraphTensorData* input_data =
      TensorDataForTensor(input.get(), op->params.dtype, device, status);
  if (input_data == nil) return;
  MPSGraphTensorData* output_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (output_data == nil) return;

  RunGraph(stream, *cached, @[ gradient_data, input_data ], @[ output_data ],
           status);
}

/*** WRAPPERS AND REGISTRATION ***/

void Conv2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  Conv2D_ComputeImpl(static_cast<ConvOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Conv2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  Conv2DBackpropInput_ComputeImpl(static_cast<ConvOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Conv2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  Conv2DBackpropFilter_ComputeImpl(static_cast<ConvOp*>(kernel), ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterConv(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
                  void (*compute)(void*, TF_OpKernelContext*),
                  const char* host_memory_arg, TF_DataType dtype,
                  const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, compute, &ConvOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (host_memory_arg != nullptr) {
    // The gradient ops receive the target shape as a small int32 tensor. It is
    // read on the host to size the output allocation, so it must not be placed
    // on the device.
    TF_KernelBuilder_HostMemory(builder, host_memory_arg);
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

void RegisterMetalConvKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    RegisterConv("Conv2D", &ConvOp_Create<true>, &Conv2D_Compute,
                 /*host_memory_arg=*/nullptr, kDTypes[i],
                 std::string("MetalConv2D") + kSuffixes[i]);
    RegisterConv("Conv2DBackpropInput", &ConvOp_Create<true>,
                 &Conv2DBackpropInput_Compute, "input_sizes", kDTypes[i],
                 std::string("MetalConv2DBackpropInput") + kSuffixes[i]);
    RegisterConv("Conv2DBackpropFilter", &ConvOp_Create<true>,
                 &Conv2DBackpropFilter_Compute, "filter_sizes", kDTypes[i],
                 std::string("MetalConv2DBackpropFilter") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
