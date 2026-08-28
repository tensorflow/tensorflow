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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// ResizeBilinearGrad and ResizeNearestNeighborGrad.
//
// These were previously left to the host. Every MPSGraph resize-gradient entry
// point aborts the process on the current SDK with a channel mismatch
// assertion, and a kernel that crashes is worse than no kernel. They are
// written here as shaders instead, which is what a resize gradient is anyway:
// the transpose of the forward sampling, pushing each resized pixel's gradient
// back to the source pixels it read, with the same weights. Source pixels are
// shared between neighbours, so the accumulation is atomic over a destination
// blitted to zero.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct ResizeGradOp {
  bool align_corners = false;
  bool half_pixel_centers = false;
  bool nearest = false;
};

void* ResizeGradOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ResizeGradOp();
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "align_corners", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->align_corners = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "half_pixel_centers", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->half_pixel_centers = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void ResizeGradOp_Delete(void* kernel) {
  delete static_cast<ResizeGradOp*>(kernel);
}

// TensorFlow's CalculateResizeScale.
float ResizeScale(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1)
             ? static_cast<float>(in_size - 1) / (out_size - 1)
             : static_cast<float>(in_size) / static_cast<float>(out_size);
}

// The second input differs between the two ops: the bilinear gradient takes
// the original image, the nearest one takes just its size. Either way it is
// the original shape that the gradient has to land in.
bool OriginalShape(TF_OpKernelContext* ctx, bool nearest,
                   const std::vector<int64_t>& grad_shape,
                   std::vector<int64_t>* out, TF_Status* status) {
  ScopedTensor second;
  TF_GetInput(ctx, 1, second.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  if (!nearest) {
    *out = ShapeOf(second.get());
    if (out->size() != 4) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: ResizeBilinearGrad expects a rank-4 original "
                   "image.");
      return false;
    }
    return true;
  }
  const void* data = TF_TensorData(second.get());
  if (data == nullptr || TF_TensorElementCount(second.get()) != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: size must have two entries in host memory.");
    return false;
  }
  const int32_t* size = static_cast<const int32_t*>(data);
  if (size[0] <= 0 || size[1] <= 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the original size must be positive.");
    return false;
  }
  *out = {grad_shape[0], size[0], size[1], grad_shape[3]};
  return true;
}

void ResizeGrad_ComputeImpl(ResizeGradOp* op, TF_OpKernelContext* ctx,
                            bool nearest, TF_Status* status) {
  ScopedTensor grads;
  TF_GetInput(ctx, 0, grads.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> grad_shape = ShapeOf(grads.get());
  if (grad_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a resize gradient expects a rank-4 gradient.");
    return;
  }

  std::vector<int64_t> out_shape;
  if (!OriginalShape(ctx, nearest, grad_shape, &out_shape, status)) return;

  const int64_t out_count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 4,
                                 static_cast<size_t>(out_count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream),
                  nearest ? "tf_resize_nearest_grad_float"
                          : "tf_resize_bilinear_grad_float",
                  status);
  if (pipeline == nil) return;

  BufferSlice grad_slice, out_slice;
  if (!SliceForTensor(grads.get(), &grad_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  ResizeGradParams params;
  params.batch = static_cast<uint32_t>(grad_shape[0]);
  params.in_h = static_cast<uint32_t>(grad_shape[1]);
  params.in_w = static_cast<uint32_t>(grad_shape[2]);
  params.channels = static_cast<uint32_t>(grad_shape[3]);
  params.out_h = static_cast<uint32_t>(out_shape[1]);
  params.out_w = static_cast<uint32_t>(out_shape[2]);
  // The scale maps a resized coordinate back to an original one, so the
  // original is the numerator, exactly as in the forward pass.
  params.height_scale =
      ResizeScale(out_shape[1], grad_shape[1], op->align_corners);
  params.width_scale =
      ResizeScale(out_shape[2], grad_shape[2], op->align_corners);
  params.half_pixel = op->half_pixel_centers ? 1 : 0;
  params.align_corners = op->align_corners ? 1 : 0;
  params.count = static_cast<uint32_t>(ElementCount(grad_shape));
  params.padding0 = 0;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a resize "
                 "gradient.");
    return;
  }
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out_slice.buffer
             range:NSMakeRange(out_slice.offset,
                               static_cast<NSUInteger>(out_count) *
                                   sizeof(float))
             value:0];
  [zero endEncoding];
  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:grad_slice.buffer offset:grad_slice.offset atIndex:0];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

#define METAL_RESIZE_GRAD_COMPUTE(NAME, NEAREST)                            \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<ResizeGradOp*>(kernel);                          \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a resize gradient kernel has no state.");        \
    } else {                                                                \
      ResizeGrad_ComputeImpl(op, ctx, NEAREST, status);                     \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_RESIZE_GRAD_COMPUTE(ResizeBilinearGrad_Compute, false)
METAL_RESIZE_GRAD_COMPUTE(ResizeNearestGrad_Compute, true)

#undef METAL_RESIZE_GRAD_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &ResizeGradOp_Create,
                          compute, &ResizeGradOp_Delete);
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

void RegisterMetalResizeGradKernels() {
  Register("ResizeBilinearGrad", &ResizeBilinearGrad_Compute,
           "MetalResizeBilinearGrad", {});
  // The nearest gradient is told the original size rather than shown the
  // original image, and that size determines the output shape.
  Register("ResizeNearestNeighborGrad", &ResizeNearestGrad_Compute,
           "MetalResizeNearestNeighborGrad", {"size"});
}

}  // namespace metal
}  // namespace tensorflow
