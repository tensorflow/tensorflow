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

struct PoolOp {
  SpatialParams params;
  int window_h = 1;
  int window_w = 1;
};

void* PoolOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new PoolOp();
  // Pooling has no dilations attribute.
  if (!ReadSpatialParams(ctx, /*want_dilations=*/false, &op->params, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  int32_t ksize[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksize", ksize, 4, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  const int channel_index = op->params.nhwc ? 3 : 1;
  if (ksize[0] != 1 || ksize[channel_index] != 1) {
    // Pooling across the batch or the channel axis is expressible in
    // TensorFlow and unsupported by every GPU backend; ignoring those entries
    // would silently pool over the wrong axes.
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: pooling over the batch or channel dimension is not "
                 "supported.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->window_h = ksize[SpatialHeightIndex(op->params.nhwc)];
  op->window_w = ksize[SpatialWidthIndex(op->params.nhwc)];
  TF_DeleteStatus(status);
  return op;
}

void PoolOp_Delete(void* kernel) { delete static_cast<PoolOp*>(kernel); }

MPSGraphPooling2DOpDescriptor* DescriptorFor(const PoolOp& op) {
  return [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:static_cast<NSUInteger>(op.window_w)
                   kernelHeight:static_cast<NSUInteger>(op.window_h)
                      strideInX:static_cast<NSUInteger>(op.params.stride_w)
                      strideInY:static_cast<NSUInteger>(op.params.stride_h)
                   paddingStyle:op.params.same_padding
                                    ? MPSGraphPaddingStyleTF_SAME
                                    : MPSGraphPaddingStyleTF_VALID
                     dataLayout:op.params.nhwc
                                    ? MPSGraphTensorNamedDataLayoutNHWC
                                    : MPSGraphTensorNamedDataLayoutNCHW];
}

void AppendPoolKey(const PoolOp& op, std::string* key) {
  key->append("/k").append(std::to_string(op.window_h));
  key->push_back('x');
  key->append(std::to_string(op.window_w));
  key->append("/s").append(std::to_string(op.params.stride_h));
  key->push_back('x');
  key->append(std::to_string(op.params.stride_w));
  key->append(op.params.same_padding ? "/SAME" : "/VALID");
  key->append(op.params.nhwc ? "/NHWC" : "/NCHW");
  key->append("/t").append(std::to_string(static_cast<int>(op.params.dtype)));
}

// Same rule as convolution, with the pooling window in place of the filter.
int64_t PoolOutputExtent(int64_t input, int window, int stride,
                         bool same_padding) {
  if (same_padding) return (input + stride - 1) / stride;
  if (input < window) return 0;
  return (input - window) / stride + 1;
}

void MaxPool_ComputeImpl(PoolOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: MaxPool expects a rank-4 input.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const int h_index = SpatialHeightIndex(op->params.nhwc);
  const int w_index = SpatialWidthIndex(op->params.nhwc);

  std::vector<int64_t> out_shape = in_shape;
  out_shape[h_index] = PoolOutputExtent(in_shape[h_index], op->window_h,
                                        op->params.stride_h,
                                        op->params.same_padding);
  out_shape[w_index] = PoolOutputExtent(in_shape[w_index], op->window_w,
                                        op->params.stride_w,
                                        op->params.same_padding);

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

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;

  std::string key = "MaxPool";
  AppendShapeToKey(in_shape, &key);
  AppendPoolKey(*op, &key);
  const PoolOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* source =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:source];
        [out->outputs
            addObject:[out->graph maxPooling2DWithSourceTensor:source
                                                    descriptor:DescriptorFor(
                                                                   captured)
                                                          name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* in_data =
      TensorDataForTensor(input.get(), op->params.dtype, device, status);
  if (in_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ in_data ], @[ out_data ], status);
}

void MaxPoolGrad_ComputeImpl(PoolOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  // orig_input, orig_output, grad. orig_output is not needed by MPSGraph,
  // which recomputes the argmax from the source, but TensorFlow passes it and
  // it still has to be read to keep the input indices aligned.
  ScopedTensor orig_input;
  ScopedTensor orig_output;
  ScopedTensor gradient;
  TF_GetInput(ctx, 0, orig_input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, orig_output.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, gradient.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(orig_input.get());
  const std::vector<int64_t> grad_shape = ShapeOf(gradient.get());

  int64_t element_count = 1;
  for (int64_t dim : in_shape) element_count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->params.dtype, in_shape.data(),
      static_cast<int>(in_shape.size()),
      static_cast<size_t>(element_count) * TF_DataTypeSize(op->params.dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (element_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);

  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->params.dtype, &mps_dtype, status)) return;

  std::string key = "MaxPoolGrad";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendPoolKey(*op, &key);
  const PoolOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* grad =
            [out->graph placeholderWithShape:MPSShape(grad_shape)
                                    dataType:mps_dtype
                                        name:nil];
        MPSGraphTensor* source =
            [out->graph placeholderWithShape:MPSShape(in_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:grad];
        [out->inputs addObject:source];
        [out->outputs addObject:[out->graph
                                    maxPooling2DGradientWithGradientTensor:grad
                                                              sourceTensor:source
                                                                descriptor:
                                                                    DescriptorFor(
                                                                        captured)
                                                                      name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* grad_data =
      TensorDataForTensor(gradient.get(), op->params.dtype, device, status);
  if (grad_data == nil) return;
  MPSGraphTensorData* source_data =
      TensorDataForTensor(orig_input.get(), op->params.dtype, device, status);
  if (source_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->params.dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ grad_data, source_data ], @[ out_data ], status);
}

void MaxPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<PoolOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: MaxPool kernel has no state.");
  } else {
    MaxPool_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void MaxPoolGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<PoolOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: MaxPoolGrad kernel has no state.");
  } else {
    MaxPoolGrad_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterPool(const char* op_name,
                  void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
                  const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &PoolOp_Create, compute, &PoolOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalPoolingKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    RegisterPool("MaxPool", &MaxPool_Compute, kDTypes[i],
                 std::string("MetalMaxPool") + kSuffixes[i]);
    RegisterPool("MaxPoolGrad", &MaxPoolGrad_Compute, kDTypes[i],
                 std::string("MetalMaxPoolGrad") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
