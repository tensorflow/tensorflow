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
#include <cstring>
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

// CropAndResize and its two gradients.
//
// Every box samples the image at its own fractional positions, so this is a
// gather with per-thread addressing rather than anything MPSGraph expresses;
// all three run as shaders. The two gradients scatter into tensors that
// neighbouring crop elements share, so they accumulate atomically and their
// destinations are blitted to zero in the same command buffer as the dispatch.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct CropOp {
  bool nearest = false;
  float extrapolation = 0.0f;
};

void* CropOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new CropOp();
  char method[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "method", method,
                                        sizeof(method) - 1, status);
  if (TF_GetCode(status) == TF_OK) {
    if (std::strcmp(method, "nearest") == 0) {
      op->nearest = true;
    } else if (std::strcmp(method, "bilinear") != 0) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: CropAndResize supports the bilinear and nearest "
                   "methods only.");
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
  }
  TF_SetStatus(status, TF_OK, "");
  float value = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "extrapolation_value", &value,
                                       status);
  if (TF_GetCode(status) == TF_OK) op->extrapolation = value;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void CropOp_Delete(void* kernel) { delete static_cast<CropOp*>(kernel); }

// The shapes and counts every one of the three kernels needs. `image_shape`
// is the full image shape, which the gradient reads from an argument rather
// than from a tensor.
bool FillParams(const CropOp& op, const std::vector<int64_t>& image_shape,
                const std::vector<int64_t>& crop_shape, int64_t num_boxes,
                CropResizeParams* params, TF_Status* status) {
  if (image_shape.size() != 4 || crop_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: CropAndResize expects rank-4 images and crops.");
    return false;
  }
  params->batch = static_cast<uint32_t>(image_shape[0]);
  params->in_h = static_cast<uint32_t>(image_shape[1]);
  params->in_w = static_cast<uint32_t>(image_shape[2]);
  params->depth = static_cast<uint32_t>(image_shape[3]);
  params->num_boxes = static_cast<uint32_t>(num_boxes);
  params->crop_h = static_cast<uint32_t>(crop_shape[1]);
  params->crop_w = static_cast<uint32_t>(crop_shape[2]);
  params->method_nearest = op.nearest ? 1 : 0;
  params->extrapolation = op.extrapolation;
  params->count = static_cast<uint32_t>(ElementCount(crop_shape));
  params->padding0 = 0;
  params->padding1 = 0;
  return true;
}

// Reads the two-element crop size, which arrives in host memory because it
// determines the output shape.
bool ReadCropSize(TF_Tensor* t, int64_t* h, int64_t* w, TF_Status* status) {
  const void* data = TF_TensorData(t);
  if (data == nullptr || TF_TensorElementCount(t) != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: crop_size must have two entries in host memory.");
    return false;
  }
  const int32_t* p = static_cast<const int32_t*>(data);
  *h = p[0];
  *w = p[1];
  if (*h <= 0 || *w <= 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: crop_size must be positive.");
    return false;
  }
  return true;
}

void CropAndResize_ComputeImpl(CropOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor image, boxes, box_index, crop_size;
  TF_GetInput(ctx, 0, image.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, boxes.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, box_index.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, crop_size.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  int64_t crop_h = 0, crop_w = 0;
  if (!ReadCropSize(crop_size.get(), &crop_h, &crop_w, status)) return;

  const std::vector<int64_t> image_shape = ShapeOf(image.get());
  const std::vector<int64_t> boxes_shape = ShapeOf(boxes.get());
  if (boxes_shape.size() != 2 || boxes_shape[1] != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: boxes must have shape [num_boxes, 4].");
    return;
  }
  const int64_t num_boxes = boxes_shape[0];
  const std::vector<int64_t> crop_shape = {num_boxes, crop_h, crop_w,
                                           image_shape.size() == 4
                                               ? image_shape[3]
                                               : 0};

  const int64_t count = ElementCount(crop_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, crop_shape.data(), 4,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  CropResizeParams params;
  if (!FillParams(*op, image_shape, crop_shape, num_boxes, &params, status)) {
    return;
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), "tf_crop_and_resize_float", status);
  if (pipeline == nil) return;

  BufferSlice img, box, idx, out;
  if (!SliceForTensor(image.get(), &img, status)) return;
  if (!SliceForTensor(boxes.get(), &box, status)) return;
  if (!SliceForTensor(box_index.get(), &idx, status)) return;
  if (!SliceForTensor(output.get(), &out, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for "
                 "CropAndResize.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:img.buffer offset:img.offset atIndex:0];
  [encoder setBuffer:box.buffer offset:box.offset atIndex:1];
  [encoder setBuffer:idx.buffer offset:idx.offset atIndex:2];
  [encoder setBuffer:out.buffer offset:out.offset atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

void CropAndResizeGradImage_ComputeImpl(CropOp* op, TF_OpKernelContext* ctx,
                                        TF_Status* status) {
  ScopedTensor grads, boxes, box_index, image_size;
  TF_GetInput(ctx, 0, grads.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, boxes.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, box_index.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, image_size.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const void* size_data = TF_TensorData(image_size.get());
  if (size_data == nullptr || TF_TensorElementCount(image_size.get()) != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: image_size must have four entries in host memory.");
    return;
  }
  const int32_t* size = static_cast<const int32_t*>(size_data);
  const std::vector<int64_t> image_shape = {size[0], size[1], size[2],
                                            size[3]};
  const std::vector<int64_t> crop_shape = ShapeOf(grads.get());
  const std::vector<int64_t> boxes_shape = ShapeOf(boxes.get());
  if (boxes_shape.size() != 2 || boxes_shape[1] != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: boxes must have shape [num_boxes, 4].");
    return;
  }

  const int64_t out_count = ElementCount(image_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, image_shape.data(), 4,
                                 static_cast<size_t>(out_count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  CropResizeParams params;
  if (!FillParams(*op, image_shape, crop_shape, boxes_shape[0], &params,
                  status)) {
    return;
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream), "tf_crop_and_resize_grad_image_float", status);
  if (pipeline == nil) return;

  BufferSlice grad, box, idx, out;
  if (!SliceForTensor(grads.get(), &grad, status)) return;
  if (!SliceForTensor(boxes.get(), &box, status)) return;
  if (!SliceForTensor(box_index.get(), &idx, status)) return;
  if (!SliceForTensor(output.get(), &out, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the "
                 "CropAndResize image gradient.");
    return;
  }
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out.buffer
             range:NSMakeRange(out.offset, static_cast<NSUInteger>(out_count) *
                                               sizeof(float))
             value:0];
  [zero endEncoding];
  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:grad.buffer offset:grad.offset atIndex:0];
    [encoder setBuffer:box.buffer offset:box.offset atIndex:1];
    [encoder setBuffer:idx.buffer offset:idx.offset atIndex:2];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

void CropAndResizeGradBoxes_ComputeImpl(CropOp* op, TF_OpKernelContext* ctx,
                                        TF_Status* status) {
  ScopedTensor grads, image, boxes, box_index;
  TF_GetInput(ctx, 0, grads.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, image.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, boxes.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, box_index.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> image_shape = ShapeOf(image.get());
  const std::vector<int64_t> crop_shape = ShapeOf(grads.get());
  const std::vector<int64_t> boxes_shape = ShapeOf(boxes.get());
  if (boxes_shape.size() != 2 || boxes_shape[1] != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: boxes must have shape [num_boxes, 4].");
    return;
  }
  const int64_t out_count = boxes_shape[0] * 4;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, boxes_shape.data(), 2,
                                 static_cast<size_t>(out_count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  CropResizeParams params;
  if (!FillParams(*op, image_shape, crop_shape, boxes_shape[0], &params,
                  status)) {
    return;
  }
  // The box gradient is only defined for bilinear sampling: the
  // nearest-neighbour position is piecewise constant in the box, so its
  // derivative is zero wherever it exists.
  params.method_nearest = 0;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream), "tf_crop_and_resize_grad_boxes_float", status);
  if (pipeline == nil) return;

  BufferSlice grad, img, box, idx, out;
  if (!SliceForTensor(grads.get(), &grad, status)) return;
  if (!SliceForTensor(image.get(), &img, status)) return;
  if (!SliceForTensor(boxes.get(), &box, status)) return;
  if (!SliceForTensor(box_index.get(), &idx, status)) return;
  if (!SliceForTensor(output.get(), &out, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the "
                 "CropAndResize box gradient.");
    return;
  }
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out.buffer
             range:NSMakeRange(out.offset, static_cast<NSUInteger>(out_count) *
                                               sizeof(float))
             value:0];
  [zero endEncoding];
  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:grad.buffer offset:grad.offset atIndex:0];
    [encoder setBuffer:img.buffer offset:img.offset atIndex:1];
    [encoder setBuffer:box.buffer offset:box.offset atIndex:2];
    [encoder setBuffer:idx.buffer offset:idx.offset atIndex:3];
    [encoder setBuffer:out.buffer offset:out.offset atIndex:4];
    [encoder setBytes:&params length:sizeof(params) atIndex:5];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

#define METAL_CROP_COMPUTE(NAME, IMPL)                                      \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<CropOp*>(kernel);                                \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a CropAndResize kernel has no state.");          \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_CROP_COMPUTE(CropAndResize_Compute, CropAndResize_ComputeImpl)
METAL_CROP_COMPUTE(CropAndResizeGradImage_Compute,
                   CropAndResizeGradImage_ComputeImpl)
METAL_CROP_COMPUTE(CropAndResizeGradBoxes_Compute,
                   CropAndResizeGradBoxes_ComputeImpl)

#undef METAL_CROP_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &CropOp_Create, compute, &CropOp_Delete);
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

void RegisterMetalCropResizeKernels() {
  // Float32 only: the two gradients accumulate with a Metal atomic, which
  // exists for float and not for half, and the forward is registered on the
  // same type so a graph cannot end up split across devices between the crop
  // and its gradient.
  Register("CropAndResize", &CropAndResize_Compute, "MetalCropAndResize",
           {"crop_size"});
  Register("CropAndResizeGradImage", &CropAndResizeGradImage_Compute,
           "MetalCropAndResizeGradImage", {"image_size"});
  Register("CropAndResizeGradBoxes", &CropAndResizeGradBoxes_Compute,
           "MetalCropAndResizeGradBoxes", {});
}

}  // namespace metal
}  // namespace tensorflow
