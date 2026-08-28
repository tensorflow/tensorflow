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

// ExtractVolumePatches.
//
// A pure gather: each output element names one voxel of the input, or the
// padding. It runs as a shader because the addressing is per-thread; there is
// no arithmetic to hand to MPSGraph.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct VolumePatchOp {
  int k[3] = {1, 1, 1};
  int stride[3] = {1, 1, 1};
  bool same_padding = false;
};

void* VolumePatchOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new VolumePatchOp();
  struct { const char* name; int* out; } lists[] = {
      {"ksizes", op->k},
      {"strides", op->stride},
  };
  for (auto& l : lists) {
    int32_t v[5] = {1, 1, 1, 1, 1};
    TF_OpKernelConstruction_GetAttrInt32List(ctx, l.name, v, 5, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
    if (v[0] != 1 || v[4] != 1) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: ExtractVolumePatches over the batch or channel "
                   "dimension is not supported.");
      TF_OpKernelConstruction_Failure(ctx, status);
      TF_DeleteStatus(status);
      delete op;
      return nullptr;
    }
    for (int i = 0; i < 3; ++i) l.out[i] = v[i + 1];
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
                 "Metal: ExtractVolumePatches supports SAME and VALID padding "
                 "only.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void VolumePatchOp_Delete(void* kernel) {
  delete static_cast<VolumePatchOp*>(kernel);
}

void VolumePatch_ComputeImpl(VolumePatchOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> in_shape = ShapeOf(input.get());
  if (in_shape.size() != 5) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: ExtractVolumePatches expects a rank-5 input.");
    return;
  }
  const int64_t channels = in_shape[4];

  int64_t out[3];
  int64_t pad[3];
  for (int i = 0; i < 3; ++i) {
    const int64_t in = in_shape[i + 1];
    const int64_t k = op->k[i];
    const int64_t s = op->stride[i];
    if (op->same_padding) {
      out[i] = (in + s - 1) / s;
      pad[i] = std::max<int64_t>(0, (out[i] - 1) * s + k - in) / 2;
    } else {
      out[i] = in < k ? 0 : (in - k) / s + 1;
      pad[i] = 0;
    }
  }
  const int64_t patch =
      static_cast<int64_t>(op->k[0]) * op->k[1] * op->k[2] * channels;
  const std::vector<int64_t> out_shape = {in_shape[0], out[0], out[1], out[2],
                                          patch};

  const int64_t count = ElementCount(out_shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 5,
                                 static_cast<size_t>(count) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream), "tf_extract_volume_patches_float", status);
  if (pipeline == nil) return;

  BufferSlice in_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  VolumePatchParams params;
  params.batch = static_cast<uint32_t>(in_shape[0]);
  params.in_d = static_cast<uint32_t>(in_shape[1]);
  params.in_h = static_cast<uint32_t>(in_shape[2]);
  params.in_w = static_cast<uint32_t>(in_shape[3]);
  params.channels = static_cast<uint32_t>(channels);
  params.out_d = static_cast<uint32_t>(out[0]);
  params.out_h = static_cast<uint32_t>(out[1]);
  params.out_w = static_cast<uint32_t>(out[2]);
  params.kd = static_cast<uint32_t>(op->k[0]);
  params.kh = static_cast<uint32_t>(op->k[1]);
  params.kw = static_cast<uint32_t>(op->k[2]);
  params.stride_d = static_cast<uint32_t>(op->stride[0]);
  params.stride_h = static_cast<uint32_t>(op->stride[1]);
  params.stride_w = static_cast<uint32_t>(op->stride[2]);
  params.pad_d = static_cast<int32_t>(pad[0]);
  params.pad_h = static_cast<int32_t>(pad[1]);
  params.pad_w = static_cast<int32_t>(pad[2]);
  params.count = static_cast<uint32_t>(count);
  params.padding0 = 0;
  params.padding1 = 0;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for "
                 "ExtractVolumePatches.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

void VolumePatch_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<VolumePatchOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: ExtractVolumePatches has no state.");
  } else {
    VolumePatch_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &VolumePatchOp_Create, compute,
      &VolumePatchOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
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

void RegisterMetalVolumePatchKernels() {
  Register("ExtractVolumePatches", &VolumePatch_Compute,
           "MetalExtractVolumePatches");
}

}  // namespace metal
}  // namespace tensorflow
