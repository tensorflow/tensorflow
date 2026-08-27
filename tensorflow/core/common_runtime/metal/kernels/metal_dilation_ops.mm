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
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_mps_graph.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Dilation2D: grayscale morphological dilation.
//
// It is a convolution with max in place of sum, which MPSGraph has no
// primitive for. Since the filter window is small and known when the graph is
// built, the maximum is unrolled: one shifted slice per filter position, each
// offset by the filter's own value, combined pairwise with maximum. That
// keeps everything in the graph and costs kh*kw slices rather than a shader.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

NSArray<NSNumber*>* ToNS(const std::vector<int64_t>& v) {
  NSMutableArray<NSNumber*>* a = [NSMutableArray array];
  for (int64_t x : v) [a addObject:@(static_cast<NSInteger>(x))];
  return a;
}

struct DilationOp {
  TF_DataType dtype = TF_FLOAT;
  int stride_h = 1, stride_w = 1;
  int rate_h = 1, rate_w = 1;
  bool same_padding = false;
};

void* DilationOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DilationOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  struct { const char* name; int* h; int* w; } lists[] = {
      {"strides", &op->stride_h, &op->stride_w},
      {"rates", &op->rate_h, &op->rate_w},
  };
  for (auto& l : lists) {
    int32_t v[4] = {1, 1, 1, 1};
    TF_OpKernelConstruction_GetAttrInt32List(ctx, l.name, v, 4, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_SetStatus(status, TF_OK, "");
      continue;
    }
    if (v[0] != 1 || v[3] != 1) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: Dilation2D over the batch or channel dimension is "
                   "not supported.");
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
    *l.h = v[1];
    *l.w = v[2];
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
                 "Metal: Dilation2D supports SAME and VALID padding only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void DilationOp_Delete(void* kernel) {
  delete static_cast<DilationOp*>(kernel);
}

void Dilation2D_ComputeImpl(DilationOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input, filter;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4 || TF_NumDims(filter.get()) != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Dilation2D expects a rank-4 input and rank-3 filter.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> f_shape = ShapeOf(filter.get());
  const int64_t kh = f_shape[0], kw = f_shape[1];
  const int64_t channels = f_shape[2];
  if (channels != in_shape[3]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Dilation2D filter and input channels differ.");
    return;
  }

  const int64_t eff_h = (kh - 1) * op->rate_h + 1;
  const int64_t eff_w = (kw - 1) * op->rate_w + 1;
  std::vector<int64_t> out_shape = in_shape;
  if (op->same_padding) {
    out_shape[1] = (in_shape[1] + op->stride_h - 1) / op->stride_h;
    out_shape[2] = (in_shape[2] + op->stride_w - 1) / op->stride_w;
  } else {
    out_shape[1] = in_shape[1] < eff_h
                       ? 0
                       : (in_shape[1] - eff_h) / op->stride_h + 1;
    out_shape[2] = in_shape[2] < eff_w
                       ? 0
                       : (in_shape[2] - eff_w) / op->stride_w + 1;
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(), 4,
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  MPSDataType mps_dtype;
  if (!MPSTypeFor(op->dtype, &mps_dtype, status)) return;

  // SAME padding centres the window; the low-side pad is what TensorFlow uses.
  const int64_t pad_h =
      op->same_padding
          ? std::max<int64_t>(0, (out_shape[1] - 1) * op->stride_h + eff_h -
                                     in_shape[1])
          : 0;
  const int64_t pad_w =
      op->same_padding
          ? std::max<int64_t>(0, (out_shape[2] - 1) * op->stride_w + eff_w -
                                     in_shape[2])
          : 0;
  const int64_t pad_top = pad_h / 2;
  const int64_t pad_left = pad_w / 2;

  std::string key = "Dilation2D";
  AppendShapeToKey(in_shape, &key);
  AppendShapeToKey(f_shape, &key);
  key.append("/s").append(std::to_string(op->stride_h)).push_back('x');
  key.append(std::to_string(op->stride_w));
  key.append("/r").append(std::to_string(op->rate_h)).push_back('x');
  key.append(std::to_string(op->rate_w));
  key.append(op->same_padding ? "/SAME" : "/VALID");
  key.append("/t").append(std::to_string(static_cast<int>(op->dtype)));

  const DilationOp captured = *op;
  const std::vector<int64_t> padded_shape = {
      in_shape[0], in_shape[1] + pad_h, in_shape[2] + pad_w, in_shape[3]};

  const CachedGraph* cached = LookupOrBuildGraph(
      key,
      ^(CachedGraph* out) {
        MPSGraph* g = out->graph;
        MPSGraphTensor* x = [g placeholderWithShape:MPSShape(in_shape)
                                           dataType:mps_dtype
                                               name:nil];
        MPSGraphTensor* f = [g placeholderWithShape:MPSShape(f_shape)
                                           dataType:mps_dtype
                                               name:nil];
        // Padding with a very negative value rather than zero: this is a
        // maximum, so a zero pad would win wherever the real data is negative.
        MPSGraphTensor* padded = x;
        if (pad_h > 0 || pad_w > 0) {
          std::vector<int64_t> lo = {0, pad_top, pad_left, 0};
          std::vector<int64_t> hi = {0, pad_h - pad_top, pad_w - pad_left, 0};
          padded = [g padTensor:x
                withPaddingMode:MPSGraphPaddingModeConstant
                    leftPadding:ToNS(lo)
                   rightPadding:ToNS(hi)
                  constantValue:-3.0e38
                           name:nil];
        }

        MPSGraphTensor* best = nil;
        for (int64_t ky = 0; ky < kh; ++ky) {
          for (int64_t kx = 0; kx < kw; ++kx) {
            // One shifted view per filter position, strided like the output.
            MPSGraphTensor* shifted =
                [g sliceTensor:padded
                        starts:@[ @0, @(ky * captured.rate_h),
                                  @(kx * captured.rate_w), @0 ]
                          ends:@[
                            @(padded_shape[0]),
                            @(ky * captured.rate_h +
                              (out_shape[1] - 1) * captured.stride_h + 1),
                            @(kx * captured.rate_w +
                              (out_shape[2] - 1) * captured.stride_w + 1),
                            @(padded_shape[3])
                          ]
                       strides:@[ @1, @(captured.stride_h),
                                  @(captured.stride_w), @1 ]
                          name:nil];
            // The filter value for this position broadcasts over the batch and
            // spatial axes.
            MPSGraphTensor* fv =
                [g sliceTensor:f dimension:0 start:ky length:1 name:nil];
            fv = [g sliceTensor:fv dimension:1 start:kx length:1 name:nil];
            fv = [g reshapeTensor:fv
                        withShape:@[ @1, @1, @1, @(channels) ]
                             name:nil];
            MPSGraphTensor* candidate =
                [g additionWithPrimaryTensor:shifted
                             secondaryTensor:fv
                                        name:nil];
            best = best == nil
                       ? candidate
                       : [g maximumWithPrimaryTensor:best
                                     secondaryTensor:candidate
                                                name:nil];
          }
        }
        [out->inputs addObject:x];
        [out->inputs addObject:f];
        [out->outputs addObject:best];
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

void Dilation2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<DilationOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Dilation2D kernel has no state.");
  } else {
    Dilation2D_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

/*** GRADIENTS ***/

// Both gradients replay the forward argmax and scatter one output gradient
// into the winning position. Several output positions routinely win the same
// input or filter element, so this is a scatter-add, which MPSGraph has no
// operation for at this shape; it runs as a shader with atomic accumulation
// instead. That also means the destination has to start at zero, hence the
// blit fill ahead of the dispatch, in the same command buffer so the ordering
// is the queue's rather than the host's.

bool DilationGeometry(DilationOp* op, const std::vector<int64_t>& in_shape,
                      const std::vector<int64_t>& f_shape,
                      const std::vector<int64_t>& out_shape,
                      DilationParams* params, TF_Status* status) {
  const int64_t kh = f_shape[0], kw = f_shape[1];
  const int64_t eff_h = (kh - 1) * op->rate_h + 1;
  const int64_t eff_w = (kw - 1) * op->rate_w + 1;
  const int64_t pad_h =
      op->same_padding
          ? std::max<int64_t>(
                0, (out_shape[1] - 1) * op->stride_h + eff_h - in_shape[1])
          : 0;
  const int64_t pad_w =
      op->same_padding
          ? std::max<int64_t>(
                0, (out_shape[2] - 1) * op->stride_w + eff_w - in_shape[2])
          : 0;
  params->batch = static_cast<uint32_t>(in_shape[0]);
  params->in_h = static_cast<uint32_t>(in_shape[1]);
  params->in_w = static_cast<uint32_t>(in_shape[2]);
  params->channels = static_cast<uint32_t>(in_shape[3]);
  params->out_h = static_cast<uint32_t>(out_shape[1]);
  params->out_w = static_cast<uint32_t>(out_shape[2]);
  params->kh = static_cast<uint32_t>(kh);
  params->kw = static_cast<uint32_t>(kw);
  params->stride_h = static_cast<uint32_t>(op->stride_h);
  params->stride_w = static_cast<uint32_t>(op->stride_w);
  params->rate_h = static_cast<uint32_t>(op->rate_h);
  params->rate_w = static_cast<uint32_t>(op->rate_w);
  params->pad_top = static_cast<int32_t>(pad_h / 2);
  params->pad_left = static_cast<int32_t>(pad_w / 2);
  params->count = static_cast<uint32_t>(ElementCount(out_shape));
  params->padding0 = 0;
  if (f_shape[2] != in_shape[3]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Dilation2D filter and input channels differ.");
    return false;
  }
  return true;
}

// `want_filter` selects which of the two tensors the gradient lands in; the
// window search is identical either way.
void DilationGrad_ComputeImpl(DilationOp* op, TF_OpKernelContext* ctx,
                              bool want_filter, TF_Status* status) {
  ScopedTensor input, filter, grad;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, filter.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumDims(input.get()) != 4 || TF_NumDims(filter.get()) != 3 ||
      TF_NumDims(grad.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the Dilation2D gradients expect a rank-4 input, a "
                 "rank-3 filter and a rank-4 out_backprop.");
    return;
  }

  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  const std::vector<int64_t> f_shape = ShapeOf(filter.get());
  const std::vector<int64_t> out_shape = ShapeOf(grad.get());
  DilationParams params;
  if (!DilationGeometry(op, in_shape, f_shape, out_shape, &params, status)) {
    return;
  }

  const std::vector<int64_t>& result_shape = want_filter ? f_shape : in_shape;
  const int64_t result_count = ElementCount(result_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, result_shape.data(),
      static_cast<int>(result_shape.size()),
      static_cast<size_t>(result_count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (result_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream),
      want_filter ? "tf_dilation_backprop_filter_float"
                  : "tf_dilation_backprop_input_float",
      status);
  if (pipeline == nil) return;

  BufferSlice in_slice, f_slice, g_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(filter.get(), &f_slice, status)) return;
  if (!SliceForTensor(grad.get(), &g_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a Dilation2D "
                 "gradient.");
    return;
  }

  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out_slice.buffer
             range:NSMakeRange(out_slice.offset,
                               static_cast<NSUInteger>(result_count) *
                                   TF_DataTypeSize(op->dtype))
             value:0];
  [zero endEncoding];

  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
    [encoder setBuffer:f_slice.buffer offset:f_slice.offset atIndex:1];
    [encoder setBuffer:g_slice.buffer offset:g_slice.offset atIndex:2];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

void DilationBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<DilationOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: Dilation2DBackpropInput kernel has no state.");
  } else {
    DilationGrad_ComputeImpl(op, ctx, /*want_filter=*/false, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void DilationBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<DilationOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: Dilation2DBackpropFilter kernel has no state.");
  } else {
    DilationGrad_ComputeImpl(op, ctx, /*want_filter=*/true, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalDilationKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  for (int i = 0; i < 2; ++i) {
    TF_Status* status = TF_NewStatus();
    TF_KernelBuilder* builder =
        TF_NewKernelBuilder("Dilation2D", kMetalDeviceType, &DilationOp_Create,
                            &Dilation2D_Compute, &DilationOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", kDTypes[i], status);
    const std::string name = std::string("MetalDilation2D") + kSuffixes[i];
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

  // Float32 only: the scatter-add is a Metal atomic, and Metal has no atomic
  // add on half. Registering half here would silently accumulate wrong, so it
  // is left to fall back rather than pretended.
  struct { const char* op_name; void (*compute)(void*, TF_OpKernelContext*); }
      grads[] = {
          {"Dilation2DBackpropInput", &DilationBackpropInput_Compute},
          {"Dilation2DBackpropFilter", &DilationBackpropFilter_Compute},
      };
  for (auto& g : grads) {
    TF_Status* status = TF_NewStatus();
    TF_KernelBuilder* builder =
        TF_NewKernelBuilder(g.op_name, kMetalDeviceType, &DilationOp_Create,
                            g.compute, &DilationOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
    const std::string name = std::string("Metal") + g.op_name + "Float";
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
}

}  // namespace metal
}  // namespace tensorflow
