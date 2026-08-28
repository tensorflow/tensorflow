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

// Depthwise convolution with both gradients, and the pooling variants whose
// window comes in as a tensor rather than an attribute.
//
// DepthwiseConv2dNative is what the MobileNet family is built from, so its
// absence pushed a whole class of models onto the host. TensorFlow's filter
// layout, [height, width, in_channels, channel_multiplier], is MPSGraph's
// HWIO for the depthwise descriptor, where the O index is documented as the
// channel multiplier. No repacking is needed.

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

struct DepthwiseOp {
  SpatialParams params;
};

void* DepthwiseOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DepthwiseOp();
  if (!ReadSpatialParams(ctx, /*want_dilations=*/true, &op->params, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void DepthwiseOp_Delete(void* kernel) {
  delete static_cast<DepthwiseOp*>(kernel);
}

MPSGraphDepthwiseConvolution2DOpDescriptor* DepthwiseDescriptor(
    const SpatialParams& p) {
  return [MPSGraphDepthwiseConvolution2DOpDescriptor
      descriptorWithStrideInX:static_cast<NSUInteger>(p.stride_w)
                    strideInY:static_cast<NSUInteger>(p.stride_h)
              dilationRateInX:static_cast<NSUInteger>(p.dilation_w)
              dilationRateInY:static_cast<NSUInteger>(p.dilation_h)
                  paddingLeft:0
                 paddingRight:0
                   paddingTop:0
                paddingBottom:0
                 paddingStyle:p.same_padding ? MPSGraphPaddingStyleTF_SAME
                                             : MPSGraphPaddingStyleTF_VALID
                   dataLayout:p.nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                     : MPSGraphTensorNamedDataLayoutNCHW
                weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
}

void AppendParams(const SpatialParams& p, std::string* key) {
  key->append("/s").append(std::to_string(p.stride_h)).push_back('x');
  key->append(std::to_string(p.stride_w));
  key->append("/d").append(std::to_string(p.dilation_h)).push_back('x');
  key->append(std::to_string(p.dilation_w));
  key->append(p.same_padding ? "/SAME" : "/VALID");
  key->append(p.nhwc ? "/NHWC" : "/NCHW");
  key->append("/t").append(std::to_string(static_cast<int>(p.dtype)));
}

int64_t ConvExtent(int64_t input, int64_t filter, int stride, int dilation,
                   bool same) {
  if (same) return (input + stride - 1) / stride;
  const int64_t eff = (filter - 1) * dilation + 1;
  return input < eff ? 0 : (input - eff) / stride + 1;
}

bool ShapeFromHost(TF_Tensor* t, std::vector<int64_t>* out, const char* what,
                   TF_Status* status) {
  if (!ReadHostVector(t, out, status)) return false;
  if (out->size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 (std::string("Metal: ") + what + " must have four entries.")
                     .c_str());
    return false;
  }
  return true;
}

/*** DEPTHWISE FORWARD ***/

void Depthwise_ComputeImpl(DepthwiseOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor input, filter;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4 || TF_NumDims(filter.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: DepthwiseConv2dNative expects rank-4 inputs.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> f_shape = ShapeOf(filter.get());
  const auto& p = op->params;
  const int h = SpatialHeightIndex(p.nhwc);
  const int w = SpatialWidthIndex(p.nhwc);
  const int c = p.nhwc ? 3 : 1;

  std::vector<int64_t> out_shape = in_shape;
  out_shape[h] = ConvExtent(in_shape[h], f_shape[0], p.stride_h, p.dilation_h,
                            p.same_padding);
  out_shape[w] = ConvExtent(in_shape[w], f_shape[1], p.stride_w, p.dilation_w,
                            p.same_padding);
  // Each input channel produces channel_multiplier outputs.
  out_shape[c] = f_shape[2] * f_shape[3];

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, p.dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(p.dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(p.dtype, &mps_dtype, status)) return;

  std::string key = "DepthwiseConv2d";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(f_shape, &key);
  AppendParams(p, &key);
  const SpatialParams captured = p;

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
                                    depthwiseConvolution2DWithSourceTensor:x
                                                             weightsTensor:f
                                                                descriptor:
                                                                    DepthwiseDescriptor(
                                                                        captured)
                                                                      name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* x_data =
      TensorDataForTensor(input.get(), p.dtype, device, status);
  if (x_data == nil) return;
  MPSGraphTensorData* f_data =
      TensorDataForTensor(filter.get(), p.dtype, device, status);
  if (f_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), p.dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ x_data, f_data ], @[ o_data ], status);
}

/*** DEPTHWISE GRADIENTS ***/

template <bool kWeights>
void DepthwiseGrad_ComputeImpl(DepthwiseOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  // BackpropInput: (input_sizes, filter, out_backprop)
  // BackpropFilter: (input, filter_sizes, out_backprop)
  ScopedTensor a, b, grad;
  TF_GetInput(ctx, 0, a.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, b.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> target_shape;
  if (!ShapeFromHost(kWeights ? b.get() : a.get(), &target_shape,
                     kWeights ? "filter_sizes" : "input_sizes", status)) {
    return;
  }
  const std::vector<int64_t> other_shape =
      ShapeOf(kWeights ? a.get() : b.get());
  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  const auto& p = op->params;

  const int64_t count = ElementCount(target_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, p.dtype, target_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(p.dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(p.dtype, &mps_dtype, status)) return;

  std::string key = kWeights ? "DepthwiseFilterGrad" : "DepthwiseDataGrad";
  AppendShapeToKey(target_shape, &key);
  AppendShapeToKey(other_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  AppendParams(p, &key);
  const SpatialParams captured = p;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraphTensor* dy = [out->graph placeholderWithShape:MPSShape(grad_shape)
                                                     dataType:mps_dtype
                                                         name:nil];
        MPSGraphTensor* other =
            [out->graph placeholderWithShape:MPSShape(other_shape)
                                    dataType:mps_dtype
                                        name:nil];
        [out->inputs addObject:dy];
        [out->inputs addObject:other];
        // The gradients are the derivative of the forward convolution rather
        // than MPSGraph's own gradient entry points.
        //
        // depthwiseConvolution2DDataGradient... and its weights counterpart
        // are correct only when the channel multiplier is one. With a filter
        // of [3, 3, 3, 4] both were wrong by more than an order of magnitude,
        // while the forward convolution with that same filter matches the CPU
        // exactly. Differentiating the forward pass therefore gives the right
        // answer for every multiplier and cannot disagree with the forward
        // this backend actually computes.
        //
        // A convolution is bilinear, so the derivative with respect to one
        // operand does not depend on the value of that operand. The operand
        // this kernel was not given is a zero constant of the right shape,
        // which is enough to build the graph and contributes nothing to the
        // result.
        MPSGraphTensor* x =
            kWeights ? other
                     : [out->graph constantWithScalar:0.0
                                                shape:MPSShape(target_shape)
                                             dataType:mps_dtype];
        MPSGraphTensor* w =
            kWeights ? [out->graph constantWithScalar:0.0
                                                shape:MPSShape(target_shape)
                                             dataType:mps_dtype]
                     : other;
        MPSGraphTensor* forward = [out->graph
            depthwiseConvolution2DWithSourceTensor:x
                                     weightsTensor:w
                                        descriptor:DepthwiseDescriptor(
                                                       captured)
                                              name:nil];
        MPSGraphTensor* loss = [out->graph
            reductionSumWithTensor:[out->graph
                                       multiplicationWithPrimaryTensor:forward
                                                       secondaryTensor:dy
                                                                  name:nil]
                              axes:nil
                              name:nil];
        MPSGraphTensor* wanted = kWeights ? w : x;
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grads =
            [out->graph gradientForPrimaryTensor:loss
                                     withTensors:@[ wanted ]
                                            name:nil];
        [out->outputs addObject:grads[wanted]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), p.dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* other_data = TensorDataForTensor(
      kWeights ? a.get() : b.get(), p.dtype, device, status);
  if (other_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), p.dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data, other_data ], @[ o_data ], status);
}

/*** AVERAGE POOLING GRADIENT ***/

struct PoolGradOp {
  SpatialParams params;
  int window_h = 1;
  int window_w = 1;
  bool window_from_attr = true;
};

void* PoolGradOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new PoolGradOp();
  if (!ReadSpatialParams(ctx, /*want_dilations=*/false, &op->params, status)) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  int32_t ksize[4] = {1, 1, 1, 1};
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksize", ksize, 4, status);
  if (TF_GetCode(status) != TF_OK) {
    // MaxPoolV2 and MaxPoolGradV2 take the window as a tensor instead.
    TF_SetStatus(status, TF_OK, "");
    op->window_from_attr = false;
  } else {
    op->window_h = ksize[SpatialHeightIndex(op->params.nhwc)];
    op->window_w = ksize[SpatialWidthIndex(op->params.nhwc)];
  }
  TF_DeleteStatus(status);
  return op;
}

void PoolGradOp_Delete(void* kernel) {
  delete static_cast<PoolGradOp*>(kernel);
}

MPSGraphPooling2DOpDescriptor* PoolDescriptor(const SpatialParams& p, int kh,
                                              int kw) {
  return [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:static_cast<NSUInteger>(kw)
                   kernelHeight:static_cast<NSUInteger>(kh)
                      strideInX:static_cast<NSUInteger>(p.stride_w)
                      strideInY:static_cast<NSUInteger>(p.stride_h)
                   paddingStyle:p.same_padding ? MPSGraphPaddingStyleTF_SAME
                                               : MPSGraphPaddingStyleTF_VALID
                     dataLayout:p.nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                       : MPSGraphTensorNamedDataLayoutNCHW];
}

void AvgPoolGrad_ComputeImpl(PoolGradOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor shape_t, grad;
  TF_GetInput(ctx, 0, shape_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> in_shape;
  if (!ShapeFromHost(shape_t.get(), &in_shape, "orig_input_shape", status)) {
    return;
  }
  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  const auto& p = op->params;

  const int64_t count = ElementCount(in_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, p.dtype, in_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(p.dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(p.dtype, &mps_dtype, status)) return;

  std::string key = "AvgPoolGrad";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(grad_shape, &key);
  key.append("/k").append(std::to_string(op->window_h)).push_back('x');
  key.append(std::to_string(op->window_w));
  AppendParams(p, &key);
  const SpatialParams captured = p;
  const int kh = op->window_h;
  const int kw = op->window_w;

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* dy = [g placeholderWithShape:MPSShape(grad_shape)
                                            dataType:mps_dtype
                                                name:nil];
        // MPSGraph wants the forward source, but an average pool spreads its
        // gradient evenly regardless of the values it saw, so a zero tensor of
        // the right shape carries all the information the gradient needs.
        MPSGraphTensor* src = [g constantWithScalar:0.0
                                              shape:MPSShape(in_shape)
                                           dataType:mps_dtype];
        [out->inputs addObject:dy];
        [out->outputs addObject:[g avgPooling2DGradientWithGradientTensor:dy
                                                             sourceTensor:src
                                                               descriptor:
                                                                   PoolDescriptor(
                                                                       captured,
                                                                       kh, kw)
                                                                     name:nil]];
      },
      status);
  if (cached == nullptr) return;

  MPSGraphTensorData* g_data =
      TensorDataForTensor(grad.get(), p.dtype, device, status);
  if (g_data == nil) return;
  MPSGraphTensorData* o_data =
      TensorDataForTensor(output.get(), p.dtype, device, status);
  if (o_data == nil) return;
  RunGraph(stream, *cached, @[ g_data ], @[ o_data ], status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_COMPUTE(NAME, STATE, IMPL)                                      \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<STATE*>(kernel);                                   \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");       \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_COMPUTE(Depthwise_Compute, DepthwiseOp, Depthwise_ComputeImpl)
METAL_COMPUTE(DepthwiseDataGrad_Compute, DepthwiseOp,
              DepthwiseGrad_ComputeImpl<false>)
METAL_COMPUTE(DepthwiseFilterGrad_Compute, DepthwiseOp,
              DepthwiseGrad_ComputeImpl<true>)
METAL_COMPUTE(AvgPoolGrad_Compute, PoolGradOp, AvgPoolGrad_ComputeImpl)

#undef METAL_COMPUTE

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), void (*destroy)(void*),
              TF_DataType dtype, const std::string& name,
              const char* host_arg) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, create, compute, destroy);
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

void RegisterMetalDepthwiseKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    const TF_DataType t = kDTypes[i];
    const std::string s = kSuffixes[i];
    Register("DepthwiseConv2dNative", &DepthwiseOp_Create, &Depthwise_Compute,
             &DepthwiseOp_Delete, t, "MetalDepthwiseConv2dNative" + s, nullptr);
    Register("DepthwiseConv2dNativeBackpropInput", &DepthwiseOp_Create,
             &DepthwiseDataGrad_Compute, &DepthwiseOp_Delete, t,
             "MetalDepthwiseConv2dNativeBackpropInput" + s, "input_sizes");
    Register("DepthwiseConv2dNativeBackpropFilter", &DepthwiseOp_Create,
             &DepthwiseFilterGrad_Compute, &DepthwiseOp_Delete, t,
             "MetalDepthwiseConv2dNativeBackpropFilter" + s, "filter_sizes");
    Register("AvgPoolGrad", &PoolGradOp_Create, &AvgPoolGrad_Compute,
             &PoolGradOp_Delete, t, "MetalAvgPoolGrad" + s,
             "orig_input_shape");
  }
}

}  // namespace metal
}  // namespace tensorflow
