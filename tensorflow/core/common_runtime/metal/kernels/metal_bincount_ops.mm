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

// Bincount and DenseBincount.
//
// The output length is an input rather than a shape, so it is read on the
// host; everything else stays on the device. Counting is a scatter-add into a
// bin many values share, so it runs as a shader with atomic accumulation
// rather than through MPSGraph, and the destination is blitted to zero in the
// same command buffer as the dispatch.
//
// Values outside [0, size) are dropped rather than rejected, matching the CPU
// kernel.

struct BincountOp {
  TF_DataType dtype = TF_FLOAT;
  bool binary_output = false;
};

void* BincountOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new BincountOp();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  TF_Bool binary = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "binary_output", &binary, status);
  if (TF_GetCode(status) != TF_OK) TF_SetStatus(status, TF_OK, "");
  op->binary_output = binary != 0;
  TF_DeleteStatus(status);
  return op;
}

void BincountOp_Delete(void* kernel) { delete static_cast<BincountOp*>(kernel); }

// Reads the single scalar that gives the number of bins.
bool ReadSize(TF_Tensor* t, int64_t* out, TF_Status* status) {
  const void* data = TF_TensorData(t);
  if (data == nullptr || TF_TensorElementCount(t) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the bin count must be a scalar in host memory.");
    return false;
  }
  const TF_DataType dtype = TF_TensorType(t);
  if (dtype == TF_INT32) {
    *out = *static_cast<const int32_t*>(data);
  } else if (dtype == TF_INT64) {
    *out = *static_cast<const int64_t*>(data);
  } else {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the bin count must be int32 or int64.");
    return false;
  }
  if (*out < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the bin count must not be negative.");
    return false;
  }
  return true;
}

void Bincount_ComputeImpl(BincountOp* op, TF_OpKernelContext* ctx, bool dense,
                          TF_Status* status) {
  ScopedTensor values, size_t_tensor, weights;
  TF_GetInput(ctx, 0, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, size_t_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, weights.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  int64_t size = 0;
  if (!ReadSize(size_t_tensor.get(), &size, status)) return;

  const std::vector<int64_t> in_shape = ShapeOf(values.get());
  const int64_t value_count = NumElements(values.get());
  const int64_t weight_count = NumElements(weights.get());
  if (weight_count != 0 && weight_count != value_count) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: bincount weights must be empty or match the input.");
    return;
  }

  // DenseBincount gives each row of a rank-2 input its own stretch of bins;
  // everything else produces one flat histogram.
  const bool two_dimensional = dense && in_shape.size() == 2;
  if (dense && in_shape.size() > 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: DenseBincount expects a rank-1 or rank-2 input.");
    return;
  }
  std::vector<int64_t> out_shape;
  if (two_dimensional) {
    out_shape = {in_shape[0], size};
  } else {
    out_shape = {size};
  }
  int64_t out_count = 1;
  for (int64_t d : out_shape) out_count *= d;

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, out_shape.data(),
      static_cast<int>(out_shape.size()),
      static_cast<size_t>(out_count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (out_count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  const TF_DataType index_dtype = TF_TensorType(values.get());
  const std::string fn = std::string("tf_bincount_") +
                         (op->dtype == TF_FLOAT ? "float" : "int") +
                         (index_dtype == TF_INT64 ? "_i64" : "_i32");
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), fn.c_str(), status);
  if (pipeline == nil) return;

  BufferSlice values_slice, weights_slice, out_slice;
  if (!SliceForTensor(values.get(), &values_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;
  // With no weights there is nothing to bind, and Metal still wants a buffer
  // at that index; the values themselves stand in and go unread.
  if (weight_count > 0) {
    if (!SliceForTensor(weights.get(), &weights_slice, status)) return;
  } else {
    weights_slice = values_slice;
  }

  BincountParams params;
  params.count = static_cast<uint32_t>(value_count);
  params.size = static_cast<uint32_t>(size);
  params.row_len =
      two_dimensional ? static_cast<uint32_t>(in_shape[1]) : 0;
  params.binary = op->binary_output ? 1 : 0;
  params.has_weights = weight_count > 0 ? 1 : 0;
  params.padding0 = 0;
  params.padding1 = 0;
  params.padding2 = 0;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a bin count.");
    return;
  }
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out_slice.buffer
             range:NSMakeRange(out_slice.offset,
                               static_cast<NSUInteger>(out_count) *
                                   TF_DataTypeSize(op->dtype))
             value:0];
  [zero endEncoding];
  if (params.count > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:values_slice.buffer
                offset:values_slice.offset
               atIndex:0];
    [encoder setBuffer:weights_slice.buffer
                offset:weights_slice.offset
               atIndex:1];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    Dispatch1D(encoder, pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

void Bincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<BincountOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Bincount kernel has no state.");
  } else {
    Bincount_ComputeImpl(op, ctx, /*dense=*/false, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void DenseBincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<BincountOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: DenseBincount kernel has no state.");
  } else {
    Bincount_ComputeImpl(op, ctx, /*dense=*/true, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &BincountOp_Create, compute,
      &BincountOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
  // The bin count sizes the output, so it has to be readable on the host.
  TF_KernelBuilder_HostMemory(builder, "size");
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

void RegisterMetalBincountKernels() {
  // Float32 and int32 only: the accumulation is a Metal atomic, and Metal has
  // no atomic add on 64-bit integers or on double.
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_INT32};
  static constexpr const char* kSuffixes[] = {"Float", "Int32"};
  for (int i = 0; i < 2; ++i) {
    Register("Bincount", &Bincount_Compute, kDTypes[i],
             std::string("MetalBincount") + kSuffixes[i]);
    Register("DenseBincount", &DenseBincount_Compute, kDTypes[i],
             std::string("MetalDenseBincount") + kSuffixes[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
