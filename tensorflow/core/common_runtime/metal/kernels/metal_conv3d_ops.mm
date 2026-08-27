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

// Conv3D and its two gradients, for volumetric models.
//
// TensorFlow's NDHWC data layout and DHWIO filter layout are MPSGraph's
// defaults for the 3-D descriptor, so as with Conv2D nothing is repacked.
// Strides and dilations are five-element lists here rather than four, with the
// batch and channel entries again required to be 1.

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

struct Conv3DOp {
  TF_DataType dtype = TF_FLOAT;
  int stride_d = 1, stride_h = 1, stride_w = 1;
  int dilation_d = 1, dilation_h = 1, dilation_w = 1;
  bool same_padding = false;
  bool ndhwc = true;
};

void* Conv3DOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new Conv3DOp();
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
    TF_SetStatus(status, TF_OK, "");
    op->ndhwc = true;
  } else {
    op->ndhwc = std::strcmp(format, "NCDHW") != 0;
  }

  char padding[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding,
                                        sizeof(padding) - 1, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  if (std::strcmp(padding, "SAME") == 0) {
    op->same_padding = true;
  } else if (std::strcmp(padding, "VALID") != 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: Conv3D supports SAME and VALID padding only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }

  // Spatial entries sit at 1..3 for NDHWC and 2..4 for NCDHW.
  const int base = op->ndhwc ? 1 : 2;
  const int channel = op->ndhwc ? 4 : 1;
  int32_t strides[5] = {1, 1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", strides, 5, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  if (strides[0] != 1 || strides[channel] != 1) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: Conv3D striding over the batch or channel dimension "
                 "is not supported.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  op->stride_d = strides[base];
  op->stride_h = strides[base + 1];
  op->stride_w = strides[base + 2];

  int32_t dilations[5] = {1, 1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "dilations", dilations, 5,
                                           status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
  } else {
    op->dilation_d = dilations[base];
    op->dilation_h = dilations[base + 1];
    op->dilation_w = dilations[base + 2];
  }
  TF_DeleteStatus(status);
  return op;
}

void Conv3DOp_Delete(void* kernel) { delete static_cast<Conv3DOp*>(kernel); }

MPSGraphConvolution3DOpDescriptor* Descriptor(const Conv3DOp& op) {
  return [MPSGraphConvolution3DOpDescriptor
      descriptorWithStrideInX:static_cast<NSUInteger>(op.stride_w)
                    strideInY:static_cast<NSUInteger>(op.stride_h)
                    strideInZ:static_cast<NSUInteger>(op.stride_d)
              dilationRateInX:static_cast<NSUInteger>(op.dilation_w)
              dilationRateInY:static_cast<NSUInteger>(op.dilation_h)
              dilationRateInZ:static_cast<NSUInteger>(op.dilation_d)
                       groups:1
                 paddingStyle:op.same_padding ? MPSGraphPaddingStyleTF_SAME
                                              : MPSGraphPaddingStyleTF_VALID
                   dataLayout:op.ndhwc ? MPSGraphTensorNamedDataLayoutNDHWC
                                       : MPSGraphTensorNamedDataLayoutNCDHW
                weightsLayout:MPSGraphTensorNamedDataLayoutDHWIO];
}

void AppendKey(const Conv3DOp& op, std::string* key) {
  key->append("/s").append(std::to_string(op.stride_d)).push_back('x');
  key->append(std::to_string(op.stride_h)).push_back('x');
  key->append(std::to_string(op.stride_w));
  key->append("/d").append(std::to_string(op.dilation_d)).push_back('x');
  key->append(std::to_string(op.dilation_h)).push_back('x');
  key->append(std::to_string(op.dilation_w));
  key->append(op.same_padding ? "/SAME" : "/VALID");
  key->append(op.ndhwc ? "/NDHWC" : "/NCDHW");
  key->append("/t").append(std::to_string(static_cast<int>(op.dtype)));
}

int64_t ConvExtent(int64_t input, int64_t filter, int stride, int dilation,
                   bool same) {
  if (same) return (input + stride - 1) / stride;
  const int64_t eff = (filter - 1) * dilation + 1;
  return input < eff ? 0 : (input - eff) / stride + 1;
}

void Conv3D_ComputeImpl(Conv3DOp* op, TF_OpKernelContext* ctx,
                        TF_Status* status) {
  ScopedTensor input, filter;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 5 || TF_NumDims(filter.get()) != 5) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Conv3D expects rank-5 inputs.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> f_shape = ShapeOf(filter.get());
  const int base = op->ndhwc ? 1 : 2;
  const int channel = op->ndhwc ? 4 : 1;

  std::vector<int64_t> out_shape = in_shape;
  out_shape[base] = ConvExtent(in_shape[base], f_shape[0], op->stride_d,
                               op->dilation_d, op->same_padding);
  out_shape[base + 1] = ConvExtent(in_shape[base + 1], f_shape[1],
                                   op->stride_h, op->dilation_h,
                                   op->same_padding);
  out_shape[base + 2] = ConvExtent(in_shape[base + 2], f_shape[2],
                                   op->stride_w, op->dilation_w,
                                   op->same_padding);
  out_shape[channel] = f_shape[4];

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 5,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = "Conv3D";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(f_shape, &key);
  AppendKey(*op, &key);
  const Conv3DOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* x = [out->graph placeholderWithShape:MPSShape(in_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        MPSGraphTensor* f = [out->graph placeholderWithShape:MPSShape(f_shape)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:x];
        [out->inputs addObject:f];
        [out->outputs addObject:[out->graph
                                    convolution3DWithSourceTensor:x
                                                    weightsTensor:f
                                                       descriptor:Descriptor(
                                                                      captured)
                                                             name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), op->dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* f_data =
      TensorDataForTensor(filter.get(), op->dtype, device, status);
  if (f_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, f_data ], @[ o_data ], status);
}

template <bool kWeights>
void Conv3DGrad_ComputeImpl(Conv3DOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor a, b, grad;
  TF_GetInput(ctx, 0, a.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, b.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> target;
  if (!ReadHostVector(kWeights ? b.get() : a.get(), &target, status)) return;
  if (target.size() != 5) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Conv3D gradient sizes must have five entries.");
    return;
  }
  const std::vector<int64_t> other = ShapeOf(kWeights ? a.get() : b.get());
  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());

  const int64_t count = ElementCount(target);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, target.data(), 5,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  std::string key = kWeights ? "Conv3DFilterGrad" : "Conv3DDataGrad";
  AppendShapeToKey(target, &key);
  AppendShapeToKey(other, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendKey(*op, &key);
  const Conv3DOp captured = *op;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* dy = [out->graph placeholderWithShape:MPSShape(grad_shape)
                                                     dataType:mps_dtype
                                                         name:nil];
        MPSGraphTensor* o = [out->graph placeholderWithShape:MPSShape(other)
                                                    dataType:mps_dtype
                                                        name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:o];
        [out->outputs
            addObject:(kWeights
                           ? [out->graph
                                 convolution3DWeightsGradientWithIncomingGradientTensor:
                                     dy
                                                                          sourceTensor:o
                                                                           outputShape:
                                                                               MPSShape(
                                                                                   target)
                                                          forwardConvolutionDescriptor:
                                                              Descriptor(captured)
                                                                                  name:nil]
                           : [out->graph
                                 convolution3DDataGradientWithIncomingGradientTensor:
                                     dy
                                                                       weightsTensor:o
                                                                         outputShape:
                                                                             MPSShape(
                                                                                 target)
                                                        forwardConvolutionDescriptor:
                                                            Descriptor(captured)
                                                                                name:nil])];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), op->dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* o_data = TensorDataForTensor(
      kWeights ? a.get() : b.get(), op->dtype, device, status);
  if (o_data == nil) return;
  MPSGraphTensorData* out_data =
      TensorDataForTensor(output.get(), op->dtype, device, status);
  if (out_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, o_data ], @[ out_data ], status);
}

#define METAL_COMPUTE(NAME, IMPL)                                             \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<Conv3DOp*>(kernel);                                \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(Conv3D_Compute, Conv3D_ComputeImpl)
METAL_COMPUTE(Conv3DDataGrad_Compute, Conv3DGrad_ComputeImpl<false>)
METAL_COMPUTE(Conv3DFilterGrad_Compute, Conv3DGrad_ComputeImpl<true>)

#undef METAL_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name, const char* host_arg) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &Conv3DOp_Create, compute, &Conv3DOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (host_arg != nullptr) TF_KernelBuilder_HostMemory(builder, host_arg);
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

void RegisterMetalConv3DKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    Register("Conv3D", &Conv3D_Compute, t, "MetalConv3D" + s, nullptr);
    Register("Conv3DBackpropInputV2", &Conv3DDataGrad_Compute, t,
             "MetalConv3DBackpropInputV2" + s, "input_sizes");
    Register("Conv3DBackpropFilterV2", &Conv3DFilterGrad_Compute, t,
             "MetalConv3DBackpropFilterV2" + s, "filter_sizes");
    // The v1 spellings take the same inputs; only the op name differs.
    Register("Conv3DBackpropInput", &Conv3DDataGrad_Compute, t,
             "MetalConv3DBackpropInput" + s, "input_sizes");
    Register("Conv3DBackpropFilter", &Conv3DFilterGrad_Compute, t,
             "MetalConv3DBackpropFilter" + s, "filter_sizes");
  }
}

}  // namespace metal
}  // namespace tensorflow
