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

// Fill, ZerosLike and OnesLike.
//
// TensorFlow's DEVICE_DEFAULT registrations for these cover int32 in host
// memory only, so the float cases a training step needs, gradient buffers
// zeroed, optimiser slots initialised, have to be provided here.

struct FillOp {
  TF_DataType dtype = TF_FLOAT;
};

void* FillOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new FillOp();
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

void FillOp_Delete(void* kernel) { delete static_cast<FillOp*>(kernel); }

const char* ConstFillShader(TF_DataType dtype) {
  return dtype == TF_HALF ? "tf_fill_const_half" : "tf_fill_const_float";
}

const char* BufferFillShader(TF_DataType dtype) {
  return dtype == TF_HALF ? "tf_fill_half" : "tf_fill_float";
}

// Fills `output` with a compile-time constant.
void EncodeConstFill(SP_Stream stream, TF_Tensor* output, TF_DataType dtype,
                     float value, int64_t count, TF_Status* status) {
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), ConstFillShader(dtype), status);
  if (pipeline == nil) return;

  BufferSlice out_slice;
  if (!SliceForTensor(output, &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a fill.");
    return;
  }

  FillParams params;
  params.count = static_cast<uint32_t>(count);
  params.value = value;
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
  [encoder setBytes:&params length:sizeof(params) atIndex:1];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** FILL ***/

void Fill_ComputeImpl(FillOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor dims;
  ScopedTensor value;
  TF_GetInput(ctx, 0, dims.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  // dims is in host memory, so the target shape can be read directly.
  const int64_t rank = TF_TensorElementCount(dims.get());
  const TF_DataType index_dtype = TF_TensorType(dims.get());
  const void* dims_data = TF_TensorData(dims.get());
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (index_dtype == TF_INT32) {
      shape.push_back(static_cast<const int32_t*>(dims_data)[i]);
    } else {
      shape.push_back(static_cast<const int64_t*>(dims_data)[i]);
    }
  }

  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  // The value stays on the device: reading it on the host would mean draining
  // the stream on every Fill, which the shape ops around it do constantly.
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), BufferFillShader(op->dtype), status);
  if (pipeline == nil) return;

  BufferSlice value_slice;
  BufferSlice out_slice;
  if (!SliceForTensor(value.get(), &value_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Fill.");
    return;
  }

  FillParams params;
  params.count = static_cast<uint32_t>(count);
  params.value = 0.0f;
  params.padding0 = 0;
  params.padding1 = 0;

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
  [encoder setBuffer:value_slice.buffer offset:value_slice.offset atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** ZEROS LIKE AND ONES LIKE ***/

template <int kValue>
void Like_ComputeImpl(FillOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  EncodeConstFill(stream, output.get(), op->dtype,
                  static_cast<float>(kValue), count, status);
}

/*** WRAPPERS AND REGISTRATION ***/

#define METAL_DEFINE_FILL_COMPUTE(NAME, IMPL)                                 \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                          \
    ScopedAutoreleasePool pool;                                               \
    TF_Status* status = TF_NewStatus();                                       \
    auto* op = static_cast<FillOp*>(kernel);                                  \
    if (op == nullptr) {                                                      \
      TF_SetStatus(status, TF_INTERNAL, "Metal: fill kernel has no state.");  \
    } else {                                                                  \
      IMPL(op, ctx, status);                                                  \
    }                                                                         \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                  \
  }

METAL_DEFINE_FILL_COMPUTE(Fill_Compute, Fill_ComputeImpl)
METAL_DEFINE_FILL_COMPUTE(ZerosLike_Compute, Like_ComputeImpl<0>)
METAL_DEFINE_FILL_COMPUTE(OnesLike_Compute, Like_ComputeImpl<1>)

#undef METAL_DEFINE_FILL_COMPUTE

void RegisterFill(const char* op_name,
                  void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
                  const char* host_memory_arg, const char* index_attr,
                  TF_DataType index_dtype, const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &FillOp_Create, compute, &FillOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  if (TF_GetCode(status) == TF_OK && index_attr != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, index_attr, index_dtype, status);
  }
  if (host_memory_arg != nullptr) {
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

void RegisterMetalFillKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_HALF};
  static constexpr const char* kSuffixes[] = {"Float", "Half"};
  static constexpr TF_DataType kIndexTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kIndexSuffixes[] = {"Int32", "Int64"};

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      RegisterFill("Fill", &Fill_Compute, kDTypes[i], "dims", "index_type",
                   kIndexTypes[j],
                   std::string("MetalFill") + kSuffixes[i] + kIndexSuffixes[j]);
    }
    RegisterFill("ZerosLike", &ZerosLike_Compute, kDTypes[i], nullptr, nullptr,
                 TF_INT32, std::string("MetalZerosLike") + kSuffixes[i]);
    RegisterFill("OnesLike", &OnesLike_Compute, kDTypes[i], nullptr, nullptr,
                 TF_INT32, std::string("MetalOnesLike") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
