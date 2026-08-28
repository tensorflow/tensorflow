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

// SparseToDense and SparseTensorDenseMatMul.
//
// Both keep the sparse tensor where it is. The dense shape arrives as a small
// host-memory input, which is all either op needs to know before it starts, so
// neither has to wait for the stream: the indices and values stay on the
// device and are read there.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct SparseOp {
  bool adjoint_a = false;
  bool adjoint_b = false;
};

void* SparseOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new SparseOp();
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "adjoint_a", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->adjoint_a = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "adjoint_b", &flag, status);
  if (TF_GetCode(status) == TF_OK) op->adjoint_b = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void SparseOp_Delete(void* kernel) { delete static_cast<SparseOp*>(kernel); }

bool ReadHostVector(TF_Tensor* t, std::vector<int64_t>* out,
                    TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a host-memory shape has no data.");
    return false;
  }
  out->clear();
  for (int64_t i = 0; i < count; ++i) {
    out->push_back(TF_TensorType(t) == TF_INT64
                       ? static_cast<const int64_t*>(data)[i]
                       : static_cast<const int32_t*>(data)[i]);
  }
  return true;
}

bool Dispatch(SP_Stream stream, const char* function,
              const std::vector<BufferSlice>& buffers,
              const SparseParams& params, uint32_t threads,
              TF_Status* status) {
  if (threads == 0) return true;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), function, status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a sparse op.");
    return false;
  }
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  NSUInteger index = 0;
  for (const BufferSlice& slice : buffers) {
    [encoder setBuffer:slice.buffer offset:slice.offset atIndex:index];
    ++index;
  }
  [encoder setBytes:&params length:sizeof(params) atIndex:index];
  Dispatch1D(encoder, pipeline, threads);
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

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

void SparseToDense_ComputeImpl(SparseOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  ScopedTensor indices, shape_tensor, values, default_value;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, default_value.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> shape;
  if (!ReadHostVector(shape_tensor.get(), &shape, status)) return;
  if (shape.empty() || shape.size() > 8) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: SparseToDense handles ranks one through eight.");
    return;
  }
  const std::vector<int64_t> index_shape = ShapeOf(indices.get());
  int64_t nnz = 0;
  int64_t rank = 1;
  if (index_shape.size() == 2) {
    nnz = index_shape[0];
    rank = index_shape[1];
  } else if (index_shape.size() == 1) {
    // A vector of indices means a rank-one tensor, one index per value.
    nnz = index_shape[0];
    rank = 1;
  } else if (!index_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: sparse indices must be a vector or a matrix.");
    return;
  }
  if (rank != static_cast<int64_t>(shape.size())) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the indices do not match the dense shape's rank.");
    return;
  }

  const int64_t count = ElementCount(shape);
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, shape.data(), static_cast<int>(shape.size()),
      static_cast<size_t>(count) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  BufferSlice out_slice, default_slice, index_slice, value_slice;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;
  if (!SliceForTensor(default_value.get(), &default_slice, status)) return;
  if (!SliceForTensor(indices.get(), &index_slice, status)) return;
  if (!SliceForTensor(values.get(), &value_slice, status)) return;

  // The background first, from a device scalar the host never reads, then the
  // values on top of it.
  {
    id<MTLComputePipelineState> fill =
        PipelineFor(DeviceForStream(stream), "tf_fill_float", status);
    if (fill == nil) return;
    OrderedCommandBuffer command_buffer(stream);
    if (!command_buffer.ok()) {
      TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                   "Metal: could not create a command buffer for "
                   "SparseToDense.");
      return;
    }
    FillParams params;
    params.count = static_cast<uint32_t>(count);
    params.value = 0.0f;
    params.padding0 = 0;
    params.padding1 = 0;
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:fill];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
    [encoder setBuffer:default_slice.buffer
                offset:default_slice.offset
               atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    Dispatch1D(encoder, fill, params.count);
    [encoder endEncoding];
    command_buffer.Commit();
  }
  if (nnz == 0) return;

  SparseParams params;
  params.nnz = static_cast<uint32_t>(nnz);
  params.rank = static_cast<uint32_t>(rank);
  params.count = static_cast<uint32_t>(nnz);
  params.inner = 1;
  params.scalar_values = NumElements(values.get()) == 1 && nnz != 1 ? 1 : 0;
  params.adjoint_a = 0;
  params.adjoint_b = 0;
  params.padding0 = 0;
  for (int i = 0; i < 8; ++i) params.shape[i] = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    params.shape[i] = static_cast<uint32_t>(shape[i]);
  }

  const std::string function =
      std::string("tf_sparse_to_dense") +
      (TF_TensorType(indices.get()) == TF_INT64 ? "_i64" : "_i32");
  std::vector<BufferSlice> buffers = {index_slice, value_slice, out_slice};
  Dispatch(stream, function.c_str(), buffers, params, params.nnz, status);
}

void SparseDenseMatMul_ComputeImpl(SparseOp* op, TF_OpKernelContext* ctx,
                                   TF_Status* status) {
  ScopedTensor indices, values, shape_tensor, dense;
  TF_GetInput(ctx, 0, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, shape_tensor.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, dense.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  std::vector<int64_t> sparse_shape;
  if (!ReadHostVector(shape_tensor.get(), &sparse_shape, status)) return;
  if (sparse_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SparseTensorDenseMatMul expects a rank-2 sparse "
                 "operand.");
    return;
  }
  const std::vector<int64_t> dense_shape = ShapeOf(dense.get());
  if (dense_shape.size() != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: SparseTensorDenseMatMul expects a rank-2 dense "
                 "operand.");
    return;
  }
  const int64_t rows = op->adjoint_a ? sparse_shape[1] : sparse_shape[0];
  const int64_t contract = op->adjoint_a ? sparse_shape[0] : sparse_shape[1];
  const int64_t cols = op->adjoint_b ? dense_shape[0] : dense_shape[1];
  const int64_t dense_contract = op->adjoint_b ? dense_shape[1] : dense_shape[0];
  if (dense_contract != contract) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the two operands do not share a contracted "
                 "dimension.");
    return;
  }

  const std::vector<int64_t> out_shape = {rows, cols};
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, out_shape.data(), 2,
      static_cast<size_t>(rows * cols) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, output.get(), status)) return;
  const int64_t nnz = NumElements(values.get());
  if (nnz == 0 || rows == 0 || cols == 0) return;

  BufferSlice index_slice, value_slice, dense_slice, out_slice;
  if (!SliceForTensor(indices.get(), &index_slice, status)) return;
  if (!SliceForTensor(values.get(), &value_slice, status)) return;
  if (!SliceForTensor(dense.get(), &dense_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  SparseParams params;
  params.nnz = static_cast<uint32_t>(nnz);
  params.rank = 2;
  params.count = static_cast<uint32_t>(nnz * cols);
  params.inner = static_cast<uint32_t>(cols);
  params.scalar_values = 0;
  params.adjoint_a = op->adjoint_a ? 1 : 0;
  params.adjoint_b = op->adjoint_b ? 1 : 0;
  params.padding0 = 0;
  for (int i = 0; i < 8; ++i) params.shape[i] = 1;
  // Already swapped, so the shader's bounds check reads the same way whether
  // or not the sparse operand is transposed.
  params.shape[0] = static_cast<uint32_t>(rows);
  params.shape[1] = static_cast<uint32_t>(contract);

  const std::string function =
      std::string("tf_sparse_dense_matmul") +
      (TF_TensorType(indices.get()) == TF_INT64 ? "_i64" : "_i32");
  std::vector<BufferSlice> buffers = {index_slice, value_slice, dense_slice,
                                      out_slice};
  Dispatch(stream, function.c_str(), buffers, params, params.count, status);
}

#define METAL_SPARSE_COMPUTE(NAME, IMPL)                                    \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<SparseOp*>(kernel);                              \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a sparse kernel has no state.");                 \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_SPARSE_COMPUTE(SparseToDense_Compute, SparseToDense_ComputeImpl)
METAL_SPARSE_COMPUTE(SparseDenseMatMul_Compute, SparseDenseMatMul_ComputeImpl)

#undef METAL_SPARSE_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType index,
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &SparseOp_Create, compute, &SparseOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "Tindices", index, status);
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

void RegisterMetalSparseKernels() {
  static constexpr TF_DataType kTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kNames[] = {"Int32", "Int64"};
  for (int i = 0; i < 2; ++i) {
    // The dense shape sizes the output, so it is read on the host; the
    // indices and values never leave the device.
    Register("SparseToDense", &SparseToDense_Compute, kTypes[i],
             std::string("MetalSparseToDense") + kNames[i],
             {"output_shape"});
    Register("SparseTensorDenseMatMul", &SparseDenseMatMul_Compute, kTypes[i],
             std::string("MetalSparseTensorDenseMatMul") + kNames[i],
             {"a_shape"});
  }
}

}  // namespace metal
}  // namespace tensorflow
