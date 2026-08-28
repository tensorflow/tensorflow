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

// Betainc, the sparse and ragged bin counts, Snapshot and Empty.
//
// What these have in common is only that each is small. They are here rather
// than each in a file of its own.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct MiscOp {
  bool binary_output = false;
  bool init = false;
  TF_DataType dtype = TF_FLOAT;
};

void* MiscOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new MiscOp();
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "binary_output", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->binary_output = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "init", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->init = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DataType dtype = TF_FLOAT;
  TF_OpKernelConstruction_GetAttrType(ctx, "dtype", &dtype, status);
  if (TF_GetCode(status) == TF_OK) op->dtype = dtype;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void MiscOp_Delete(void* kernel) { delete static_cast<MiscOp*>(kernel); }

bool ZeroTensor(SP_Stream stream, TF_Tensor* tensor, TF_Status* status) {
  BufferSlice slice;
  if (!SliceForTensor(tensor, &slice, status)) return false;
  const size_t bytes = TF_TensorByteSize(tensor);
  if (bytes == 0) return true;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer to zero a tensor.");
    return false;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  [encoder fillBuffer:slice.buffer
                range:NSMakeRange(slice.offset, bytes)
                value:0];
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

/*** BETAINC ***/

void Betainc_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                         TF_Status* status) {
  ScopedTensor a, b, x;
  TF_GetInput(ctx, 0, a.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, b.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, x.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  // The op broadcasts, but only in the one way it ever needs to: an argument
  // is either the full shape or a single value.
  const int64_t counts[3] = {NumElements(a.get()), NumElements(b.get()),
                             NumElements(x.get())};
  const int64_t count = std::max({counts[0], counts[1], counts[2]});
  std::vector<int64_t> shape = ShapeOf(x.get());
  if (counts[0] == count) {
    shape = ShapeOf(a.get());
  } else if (counts[1] == count) {
    shape = ShapeOf(b.get());
  }
  for (int i = 0; i < 3; ++i) {
    if (counts[i] != count && counts[i] != 1) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: Betainc broadcasts a scalar against the full shape "
                   "only.");
      return;
    }
  }

  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), "tf_betainc_float", status);
  if (pipeline == nil) return;

  BufferSlice a_slice, b_slice, x_slice, out_slice;
  if (!SliceForTensor(a.get(), &a_slice, status)) return;
  if (!SliceForTensor(b.get(), &b_slice, status)) return;
  if (!SliceForTensor(x.get(), &x_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  BetaincParams params;
  params.count = static_cast<uint32_t>(count);
  params.a_is_scalar = counts[0] == count ? 0 : 1;
  params.b_is_scalar = counts[1] == count ? 0 : 1;
  params.x_is_scalar = counts[2] == count ? 0 : 1;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Betainc.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:a_slice.buffer offset:a_slice.offset atIndex:0];
  [encoder setBuffer:b_slice.buffer offset:b_slice.offset atIndex:1];
  [encoder setBuffer:x_slice.buffer offset:x_slice.offset atIndex:2];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** SPARSE AND RAGGED BIN COUNTS ***/

// `ragged` selects which of the two row conventions applies: a sparse tensor
// names each value's row in its first coordinate, a ragged one implies it
// through the row splits.
void RowBincount_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx, bool ragged,
                             TF_Status* status) {
  ScopedTensor first, values, size_tensor, weights;
  TF_GetInput(ctx, 0, first.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor dense_shape;
  int size_index = 2;
  if (!ragged) {
    TF_GetInput(ctx, 2, dense_shape.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    size_index = 3;
  }
  TF_GetInput(ctx, size_index, size_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, size_index + 1, weights.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const void* size_data = TF_TensorData(size_tensor.get());
  if (size_data == nullptr || TF_TensorElementCount(size_tensor.get()) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the bin count must be a scalar in host memory.");
    return;
  }
  const int64_t size =
      TF_TensorType(size_tensor.get()) == TF_INT64
          ? *static_cast<const int64_t*>(size_data)
          : *static_cast<const int32_t*>(size_data);
  if (size < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the bin count must not be negative.");
    return;
  }

  // The number of rows, and the rank of the coordinates the shader reads.
  int64_t rows = 1;
  uint32_t row_stride = 1;
  if (ragged) {
    rows = std::max<int64_t>(NumElements(first.get()) - 1, 0);
    row_stride = static_cast<uint32_t>(NumElements(first.get()));
  } else {
    const void* shape_data = TF_TensorData(dense_shape.get());
    const int64_t rank = TF_TensorElementCount(dense_shape.get());
    if (shape_data == nullptr || rank < 1) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: the dense shape must be in host memory.");
      return;
    }
    row_stride = static_cast<uint32_t>(rank);
    rows = rank > 1 ? static_cast<const int64_t*>(shape_data)[0] : 1;
  }

  std::vector<int64_t> out_shape;
  const bool two_dimensional = ragged || row_stride > 1;
  if (two_dimensional) {
    out_shape = {rows, size};
  } else {
    out_shape = {size};
  }
  const int64_t out_count = ElementCount(out_shape);

  const TF_DataType dtype = TF_TensorType(weights.get());
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(out_count) * TF_DataTypeSize(dtype), status));
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, output.get(), status)) return;
  const int64_t value_count = NumElements(values.get());
  if (out_count == 0 || value_count == 0) return;

  const int64_t weight_count = NumElements(weights.get());
  if (weight_count != 0 && weight_count != value_count) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: bincount weights must be empty or match the values.");
    return;
  }

  BufferSlice value_slice, row_slice, weight_slice, out_slice;
  if (!SliceForTensor(values.get(), &value_slice, status)) return;
  if (!SliceForTensor(first.get(), &row_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;
  if (weight_count > 0) {
    if (!SliceForTensor(weights.get(), &weight_slice, status)) return;
  } else {
    weight_slice = value_slice;
  }

  BincountParams params;
  params.count = static_cast<uint32_t>(value_count);
  params.size = static_cast<uint32_t>(size);
  params.row_len = row_stride;
  params.binary = op->binary_output ? 1 : 0;
  params.has_weights = weight_count > 0 ? 1 : 0;
  params.padding0 = 0;
  params.padding1 = 0;
  params.padding2 = 0;

  const std::string function =
      std::string(ragged ? "tf_ragged_bincount_" : "tf_sparse_bincount_") +
      (dtype == TF_FLOAT ? "float" : "int") +
      (TF_TensorType(values.get()) == TF_INT64 ? "_i64" : "_i32");
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), function.c_str(), status);
  if (pipeline == nil) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a bin count.");
    return;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:value_slice.buffer offset:value_slice.offset atIndex:0];
  [encoder setBuffer:row_slice.buffer offset:row_slice.offset atIndex:1];
  [encoder setBuffer:weight_slice.buffer offset:weight_slice.offset atIndex:2];
  [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
}

/*** SNAPSHOT AND EMPTY ***/

// A copy that the graph keeps distinct from its input so a later mutation
// cannot be seen through it.
void Snapshot_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                          TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> shape = ShapeOf(input.get());
  const TF_DataType dtype = TF_TensorType(input.get());
  const size_t bytes = TF_TensorByteSize(input.get());

  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, dtype, shape.data(),
                                 static_cast<int>(shape.size()), bytes,
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (bytes == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  BufferSlice in_slice, out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Snapshot.");
    return;
  }
  id<MTLBlitCommandEncoder> encoder =
      [command_buffer.get() blitCommandEncoder];
  [encoder copyFromBuffer:in_slice.buffer
             sourceOffset:in_slice.offset
                 toBuffer:out_slice.buffer
        destinationOffset:out_slice.offset
                     size:bytes];
  [encoder endEncoding];
  command_buffer.Commit();
}

// An allocation of the requested shape, zeroed only if asked. Leaving it
// alone is the point of the op: its caller is about to write every element.
void Empty_ComputeImpl(MiscOp* op, TF_OpKernelContext* ctx,
                       TF_Status* status) {
  ScopedTensor shape_tensor;
  TF_GetInput(ctx, 0, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const void* data = TF_TensorData(shape_tensor.get());
  const int64_t rank = TF_TensorElementCount(shape_tensor.get());
  if (data == nullptr && rank > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the shape must be in host memory.");
    return;
  }
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < rank; ++i) {
    shape.push_back(TF_TensorType(shape_tensor.get()) == TF_INT64
                        ? static_cast<const int64_t*>(data)[i]
                        : static_cast<const int32_t*>(data)[i]);
  }
  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, op->dtype, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * TF_DataTypeSize(op->dtype), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!op->init || count == 0) return;
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  ZeroTensor(stream, output.get(), status);
}

#define METAL_MISC_COMPUTE(NAME, BODY)                                      \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<MiscOp*>(kernel);                                \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");     \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_MISC_COMPUTE(Betainc_Compute, Betainc_ComputeImpl(op, ctx, status))
METAL_MISC_COMPUTE(SparseBincount_Compute,
                   RowBincount_ComputeImpl(op, ctx, /*ragged=*/false, status))
METAL_MISC_COMPUTE(RaggedBincount_Compute,
                   RowBincount_ComputeImpl(op, ctx, /*ragged=*/true, status))
METAL_MISC_COMPUTE(Snapshot_Compute, Snapshot_ComputeImpl(op, ctx, status))
METAL_MISC_COMPUTE(Empty_Compute, Empty_ComputeImpl(op, ctx, status))

#undef METAL_MISC_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name, const char* type_attr,
              TF_DataType dtype,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &MiscOp_Create, compute, &MiscOp_Delete);
  if (type_attr != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, type_attr, dtype, status);
  }
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

void RegisterMetalMisc2Kernels() {
  Register("Betainc", &Betainc_Compute, "MetalBetainc", "T", TF_FLOAT, {});
  Register("Snapshot", &Snapshot_Compute, "MetalSnapshotFloat", "T", TF_FLOAT,
           {});
  Register("Snapshot", &Snapshot_Compute, "MetalSnapshotHalf", "T", TF_HALF,
           {});
  Register("Snapshot", &Snapshot_Compute, "MetalSnapshotInt32", "T", TF_INT32,
           {});
  Register("Snapshot", &Snapshot_Compute, "MetalSnapshotInt64", "T", TF_INT64,
           {});
  // The shape sizes the allocation, so it is read on the host.
  Register("Empty", &Empty_Compute, "MetalEmptyFloat", "dtype", TF_FLOAT,
           {"shape"});
  Register("Empty", &Empty_Compute, "MetalEmptyInt32", "dtype", TF_INT32,
           {"shape"});

  // The bin count sizes the output; the values, the coordinates and the
  // weights all stay on the device.
  static constexpr TF_DataType kTypes[] = {TF_FLOAT, TF_INT32};
  static constexpr const char* kNames[] = {"Float", "Int32"};
  for (int i = 0; i < 2; ++i) {
    Register("SparseBincount", &SparseBincount_Compute,
             std::string("MetalSparseBincount") + kNames[i], "T", kTypes[i],
             {"dense_shape", "size"});
    Register("RaggedBincount", &RaggedBincount_Compute,
             std::string("MetalRaggedBincount") + kNames[i], "T", kTypes[i],
             {"size"});
  }
}

}  // namespace metal
}  // namespace tensorflow
