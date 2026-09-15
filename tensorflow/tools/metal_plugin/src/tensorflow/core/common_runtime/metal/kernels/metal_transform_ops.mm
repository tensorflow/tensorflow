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

// ImageProjectiveTransformV2 and V3.
//
// Each output pixel is projected back into the source image and read there, so
// the addressing is per-thread and the whole thing is a shader. The only
// difference between the two ops is where the fill value comes from: V2 uses
// zero, V3 takes it as an input.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct TransformOp {
  bool nearest = false;
  // 0 CONSTANT, 1 REFLECT, 2 WRAP, 3 NEAREST, matching TensorFlow's order.
  uint32_t fill_mode = 0;
};

void* TransformOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new TransformOp();

  char interpolation[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "interpolation", interpolation,
                                        sizeof(interpolation) - 1, status);
  if (TF_GetCode(status) == TF_OK) {
    if (std::strcmp(interpolation, "NEAREST") == 0) {
      op->nearest = true;
    } else if (std::strcmp(interpolation, "BILINEAR") != 0) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: the projective transform supports NEAREST and "
                   "BILINEAR interpolation only.");
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
  }
  TF_SetStatus(status, TF_OK, "");

  char fill[16] = {0};
  TF_OpKernelConstruction_GetAttrString(ctx, "fill_mode", fill,
                                        sizeof(fill) - 1, status);
  if (TF_GetCode(status) == TF_OK && fill[0] != '\0') {
    if (std::strcmp(fill, "CONSTANT") == 0) {
      op->fill_mode = 0;
    } else if (std::strcmp(fill, "REFLECT") == 0) {
      op->fill_mode = 1;
    } else if (std::strcmp(fill, "WRAP") == 0) {
      op->fill_mode = 2;
    } else if (std::strcmp(fill, "NEAREST") == 0) {
      op->fill_mode = 3;
    } else {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: unknown fill_mode for the projective transform.");
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

void TransformOp_Delete(void* kernel) {
  delete static_cast<TransformOp*>(kernel);
}

// `fill_value_index` is the input carrying the fill value, or -1 for V2, whose
// fill value is defined to be zero.
void Transform_ComputeImpl(TransformOp* op, TF_OpKernelContext* ctx,
                           int fill_value_index, TF_Status* status) {
  ScopedTensor images, transforms, output_shape;
  TF_GetInput(ctx, 0, images.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, transforms.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, output_shape.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(images.get());
  if (in_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the projective transform expects a rank-4 image.");
    return;
  }
  const std::vector<int64_t> t_shape = ShapeOf(transforms.get());
  if (t_shape.size() != 2 || t_shape[1] != 8) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: transforms must have shape [num_transforms, 8].");
    return;
  }
  if (t_shape[0] != 1 && t_shape[0] != in_shape[0]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: there must be one transform, or one per image.");
    return;
  }

  const void* shape_data = TF_TensorData(output_shape.get());
  if (shape_data == nullptr ||
      TF_TensorElementCount(output_shape.get()) != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: output_shape must have two entries in host memory.");
    return;
  }
  const int32_t* wanted = static_cast<const int32_t*>(shape_data);
  if (wanted[0] <= 0 || wanted[1] <= 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: output_shape must be positive.");
    return;
  }
  const std::vector<int64_t> out_shape = {in_shape[0], wanted[0], wanted[1],
                                          in_shape[3]};

  float fill_value = 0.0f;
  if (fill_value_index >= 0) {
    ScopedTensor fill;
    TF_GetInput(ctx, fill_value_index, fill.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    const void* data = TF_TensorData(fill.get());
    if (data == nullptr) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: fill_value has no data.");
      return;
    }
    fill_value = *static_cast<const float*>(data);
  }

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 4,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream), "tf_image_projective_transform_float", status);
  if (pipeline == nil) return;

  BufferSlice img, tfm, out;
  if (!SliceForTensor(images.get(), &img, status)) return;
  if (!SliceForTensor(transforms.get(), &tfm, status)) return;
  if (!SliceForTensor(output.get(), &out, status)) return;

  TransformParams params;
  params.batch = static_cast<uint32_t>(in_shape[0]);
  params.in_h = static_cast<uint32_t>(in_shape[1]);
  params.in_w = static_cast<uint32_t>(in_shape[2]);
  params.depth = static_cast<uint32_t>(in_shape[3]);
  params.out_h = static_cast<uint32_t>(out_shape[1]);
  params.out_w = static_cast<uint32_t>(out_shape[2]);
  params.count = static_cast<uint32_t>(count);
  params.nearest = op->nearest ? 1 : 0;
  params.fill_mode = op->fill_mode;
  params.num_transforms = static_cast<uint32_t>(t_shape[0]);
  params.fill_value = fill_value;
  params.padding0 = 0;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for the "
                 "projective transform.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:img.buffer offset:img.offset atIndex:0];
  [encoder setBuffer:tfm.buffer offset:tfm.offset atIndex:1];
  [encoder setBuffer:out.buffer offset:out.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

#define METAL_TRANSFORM_COMPUTE(NAME, FILL_INDEX)                           \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<TransformOp*>(kernel);                           \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: the transform kernel has no state.");            \
    } else {                                                                \
      Transform_ComputeImpl(op, ctx, FILL_INDEX, status);                   \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_TRANSFORM_COMPUTE(TransformV2_Compute, -1)
METAL_TRANSFORM_COMPUTE(TransformV3_Compute, 3)

#undef METAL_TRANSFORM_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &TransformOp_Create,
                          compute, &TransformOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "dtype", TF_FLOAT, status);
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

void RegisterMetalTransformKernels() {
  // output_shape sizes the result and fill_value is one scalar the shader
  // takes by value, so both are read on the host.
  Register("ImageProjectiveTransformV2", &TransformV2_Compute,
           "MetalImageProjectiveTransformV2", {"output_shape"});
  Register("ImageProjectiveTransformV3", &TransformV3_Compute,
           "MetalImageProjectiveTransformV3", {"output_shape", "fill_value"});
}

}  // namespace metal
}  // namespace tensorflow
