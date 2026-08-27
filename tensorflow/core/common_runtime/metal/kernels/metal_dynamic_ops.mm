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
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
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

// The ops whose output shape depends on the values in their inputs: Unique,
// UniqueWithCounts, DynamicPartition, DynamicStitch and its parallel form.
//
// TensorFlow allocates an output before a kernel runs its work, so a shape
// that depends on the data has to be known on the host first. On a discrete
// GPU that means a copy back across the bus; on Apple Silicon the device
// allocation is already host-visible, so the wait is for the stream to reach
// this point and the read costs nothing beyond that.
//
// The waiting is real, though, and it is why these ops are worth avoiding in
// a hot loop rather than worth hiding. Once the shape is known, the data
// movement goes back on the GPU as a gather or a scatter.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

// Waits for everything already enqueued, so the host can read what it wrote.
void WaitForStream(SP_Stream stream) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
}

// A row's width in 32-bit words, which is the unit the movement shaders copy.
bool WordsPerRow(TF_DataType dtype, int64_t elements, uint32_t* words,
                 TF_Status* status) {
  const size_t size = TF_DataTypeSize(dtype);
  if (size != 4 && size != 8) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: this op handles four- and eight-byte types only.");
    return false;
  }
  *words = static_cast<uint32_t>(elements * (size / 4));
  return true;
}

// Puts a host-built index vector where a shader can read it.
bool StageIndices(TF_OpKernelContext* ctx, const std::vector<int32_t>& indices,
                  ScopedTensor* out, TF_Status* status) {
  int64_t dims[1] = {static_cast<int64_t>(std::max<size_t>(indices.size(), 1))};
  out->reset(TF_AllocateTemp(ctx, TF_INT32, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out->get());
  if (data == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: a staged index vector has no storage.");
    return false;
  }
  if (!indices.empty()) {
    std::memcpy(data, indices.data(), indices.size() * sizeof(int32_t));
  }
  return true;
}

// Runs one gather or scatter of whole rows.
bool MoveRows(SP_Stream stream, bool scatter, const BufferSlice& data,
              const BufferSlice& indices, const BufferSlice& out,
              uint32_t rows, uint32_t words, uint32_t limit,
              TF_Status* status) {
  if (rows == 0 || words == 0) return true;
  id<MTLComputePipelineState> pipeline = PipelineFor(
      DeviceForStream(stream),
      scatter ? "tf_scatter_rows_u32" : "tf_gather_rows_u32", status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer to move rows.");
    return false;
  }
  RowMoveParams params;
  params.count = rows * words;
  params.slice = words;
  params.limit = limit;
  params.padding0 = 0;
  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:data.buffer offset:data.offset atIndex:0];
  [encoder setBuffer:indices.buffer offset:indices.offset atIndex:1];
  [encoder setBuffer:out.buffer offset:out.offset atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  Dispatch1D(encoder, pipeline, params.count);
  [encoder endEncoding];
  command_buffer.Commit();
  return true;
}

struct DynamicOp {
  TF_DataType index_dtype = TF_INT32;
  int32_t num_partitions = 1;
};

void* DynamicOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DynamicOp();
  TF_DataType out_idx = TF_INT32;
  TF_OpKernelConstruction_GetAttrType(ctx, "out_idx", &out_idx, status);
  if (TF_GetCode(status) == TF_OK) op->index_dtype = out_idx;
  TF_SetStatus(status, TF_OK, "");
  int32_t partitions = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_partitions", &partitions,
                                       status);
  if (TF_GetCode(status) == TF_OK) op->num_partitions = partitions;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void DynamicOp_Delete(void* kernel) { delete static_cast<DynamicOp*>(kernel); }

// Writes an index vector into an output tensor of either index width.
void WriteIndices(TF_Tensor* tensor, TF_DataType dtype,
                  const std::vector<int64_t>& values) {
  void* data = TF_TensorData(tensor);
  if (data == nullptr) return;
  if (dtype == TF_INT32) {
    int32_t* p = static_cast<int32_t*>(data);
    for (size_t i = 0; i < values.size(); ++i) {
      p[i] = static_cast<int32_t>(values[i]);
    }
  } else {
    int64_t* p = static_cast<int64_t*>(data);
    for (size_t i = 0; i < values.size(); ++i) p[i] = values[i];
  }
}

/*** UNIQUE ***/

// Compares elements bitwise through their own type, so that the ordering
// matches TensorFlow's, which is first-occurrence rather than sorted.
template <typename T>
void UniqueOf(const T* values, int64_t count, std::vector<int64_t>* order,
              std::vector<int64_t>* index_of, std::vector<int64_t>* counts) {
  std::unordered_map<T, int64_t> seen;
  seen.reserve(static_cast<size_t>(count));
  index_of->assign(static_cast<size_t>(count), 0);
  for (int64_t i = 0; i < count; ++i) {
    const T value = values[i];
    auto it = seen.find(value);
    if (it == seen.end()) {
      const int64_t slot = static_cast<int64_t>(order->size());
      seen.emplace(value, slot);
      order->push_back(i);
      counts->push_back(1);
      (*index_of)[i] = slot;
    } else {
      (*index_of)[i] = it->second;
      (*counts)[it->second] += 1;
    }
  }
}

void Unique_ComputeImpl(DynamicOp* op, TF_OpKernelContext* ctx, bool counts,
                        TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> shape = ShapeOf(input.get());
  if (shape.size() != 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: Unique expects a rank-1 input.");
    return;
  }
  const int64_t count = shape[0];
  const TF_DataType dtype = TF_TensorType(input.get());

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  // The output length is the number of distinct values, which cannot be known
  // before the values exist.
  WaitForStream(stream);

  const void* data = TF_TensorData(input.get());
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: Unique input has no storage.");
    return;
  }
  std::vector<int64_t> order, index_of, occurrences;
  switch (dtype) {
    case TF_FLOAT:
      UniqueOf(static_cast<const float*>(data), count, &order, &index_of,
               &occurrences);
      break;
    case TF_INT32:
      UniqueOf(static_cast<const int32_t*>(data), count, &order, &index_of,
               &occurrences);
      break;
    case TF_INT64:
      UniqueOf(static_cast<const int64_t*>(data), count, &order, &index_of,
               &occurrences);
      break;
    default:
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: Unique handles float32, int32 and int64.");
      return;
  }

  const int64_t distinct = static_cast<int64_t>(order.size());
  const std::vector<int64_t> y_shape = {distinct};
  ScopedTensor y, idx, cnt;
  y.reset(TF_AllocateOutput(ctx, 0, dtype, y_shape.data(), 1,
                            static_cast<size_t>(distinct) *
                                TF_DataTypeSize(dtype),
                            status));
  if (TF_GetCode(status) != TF_OK) return;
  idx.reset(TF_AllocateOutput(ctx, 1, op->index_dtype, shape.data(), 1,
                              static_cast<size_t>(count) *
                                  TF_DataTypeSize(op->index_dtype),
                              status));
  if (TF_GetCode(status) != TF_OK) return;
  if (counts) {
    cnt.reset(TF_AllocateOutput(ctx, 2, op->index_dtype, y_shape.data(), 1,
                                static_cast<size_t>(distinct) *
                                    TF_DataTypeSize(op->index_dtype),
                                status));
    if (TF_GetCode(status) != TF_OK) return;
  }

  WriteIndices(idx.get(), op->index_dtype, index_of);
  if (counts) WriteIndices(cnt.get(), op->index_dtype, occurrences);

  // The distinct values themselves go back through the GPU, so the copy is
  // the device's rather than a second pass over host pointers.
  if (distinct > 0) {
    std::vector<int32_t> gather(order.begin(), order.end());
    ScopedTensor staged;
    if (!StageIndices(ctx, gather, &staged, status)) return;
    BufferSlice data_slice, index_slice, out_slice;
    if (!SliceForTensor(input.get(), &data_slice, status)) return;
    if (!SliceForTensor(staged.get(), &index_slice, status)) return;
    if (!SliceForTensor(y.get(), &out_slice, status)) return;
    uint32_t words = 0;
    if (!WordsPerRow(dtype, 1, &words, status)) return;
    if (!MoveRows(stream, /*scatter=*/false, data_slice, index_slice,
                  out_slice, static_cast<uint32_t>(distinct), words,
                  static_cast<uint32_t>(count), status)) {
      return;
    }
  }
}

/*** DYNAMIC PARTITION ***/

void DynamicPartition_ComputeImpl(DynamicOp* op, TF_OpKernelContext* ctx,
                                  TF_Status* status) {
  ScopedTensor data, partitions;
  TF_GetInput(ctx, 0, data.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, partitions.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> data_shape = ShapeOf(data.get());
  const std::vector<int64_t> part_shape = ShapeOf(partitions.get());
  if (data_shape.empty() || part_shape.size() != 1 ||
      part_shape[0] != data_shape[0]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: DynamicPartition expects partitions to match the "
                 "first dimension of data.");
    return;
  }
  const int64_t rows = data_shape[0];
  int64_t row_elements = 1;
  for (size_t i = 1; i < data_shape.size(); ++i) row_elements *= data_shape[i];
  const TF_DataType dtype = TF_TensorType(data.get());

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  // Each partition's size is a count of the values in `partitions`.
  WaitForStream(stream);

  const int32_t* part = static_cast<const int32_t*>(TF_TensorData(
      partitions.get()));
  if (part == nullptr && rows > 0) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: DynamicPartition partitions have no storage.");
    return;
  }
  const int num = std::max(op->num_partitions, 1);
  std::vector<std::vector<int32_t>> members(num);
  for (int64_t i = 0; i < rows; ++i) {
    const int32_t p = part[i];
    if (p < 0 || p >= num) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: a partition index is out of range.");
      return;
    }
    members[p].push_back(static_cast<int32_t>(i));
  }

  uint32_t words = 0;
  if (!WordsPerRow(dtype, row_elements, &words, status)) return;
  BufferSlice data_slice;
  if (!SliceForTensor(data.get(), &data_slice, status)) return;

  for (int p = 0; p < num; ++p) {
    std::vector<int64_t> out_shape = data_shape;
    out_shape[0] = static_cast<int64_t>(members[p].size());
    ScopedTensor out;
    out.reset(TF_AllocateOutput(
        ctx, p, dtype, out_shape.data(), static_cast<int>(out_shape.size()),
        static_cast<size_t>(ElementCount(out_shape)) * TF_DataTypeSize(dtype),
        status));
    if (TF_GetCode(status) != TF_OK) return;
    if (members[p].empty()) continue;
    ScopedTensor staged;
    if (!StageIndices(ctx, members[p], &staged, status)) return;
    BufferSlice index_slice, out_slice;
    if (!SliceForTensor(staged.get(), &index_slice, status)) return;
    if (!SliceForTensor(out.get(), &out_slice, status)) return;
    if (!MoveRows(stream, /*scatter=*/false, data_slice, index_slice,
                  out_slice, static_cast<uint32_t>(members[p].size()), words,
                  static_cast<uint32_t>(rows), status)) {
      return;
    }
  }
}

/*** DYNAMIC STITCH ***/

void DynamicStitch_ComputeImpl(DynamicOp* op, TF_OpKernelContext* ctx,
                               TF_Status* status) {
  // The op takes n index tensors followed by n data tensors.
  const int total = TF_NumInputs(ctx);
  if (total < 2 || total % 2 != 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: DynamicStitch expects matching index and data "
                 "lists.");
    return;
  }
  const int n = total / 2;

  std::vector<ScopedTensor> indices(n), data(n);
  std::vector<std::vector<int32_t>> index_values(n);
  int64_t max_index = -1;
  int64_t row_elements = 1;
  TF_DataType dtype = TF_FLOAT;
  for (int k = 0; k < n; ++k) {
    TF_GetInput(ctx, k, indices[k].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    TF_GetInput(ctx, n + k, data[k].address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    // The indices are pinned to host memory, so the output shape is known
    // without waiting for anything.
    const int64_t count = NumElements(indices[k].get());
    const int32_t* p =
        static_cast<const int32_t*>(TF_TensorData(indices[k].get()));
    if (p == nullptr && count > 0) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Metal: DynamicStitch indices have no storage.");
      return;
    }
    index_values[k].assign(p, p + count);
    for (int64_t i = 0; i < count; ++i) {
      max_index = std::max<int64_t>(max_index, index_values[k][i]);
    }
    const std::vector<int64_t> data_shape = ShapeOf(data[k].get());
    const std::vector<int64_t> index_shape = ShapeOf(indices[k].get());
    if (data_shape.size() < index_shape.size()) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: DynamicStitch data must be at least as deep as its "
                   "indices.");
      return;
    }
    if (k == 0) {
      dtype = TF_TensorType(data[k].get());
      row_elements = 1;
      for (size_t i = index_shape.size(); i < data_shape.size(); ++i) {
        row_elements *= data_shape[i];
      }
    }
  }

  const std::vector<int64_t> first_data = ShapeOf(data[0].get());
  const std::vector<int64_t> first_index = ShapeOf(indices[0].get());
  std::vector<int64_t> out_shape = {max_index + 1};
  for (size_t i = first_index.size(); i < first_data.size(); ++i) {
    out_shape.push_back(first_data[i]);
  }

  ScopedTensor out;
  out.reset(TF_AllocateOutput(
      ctx, 0, dtype, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(ElementCount(out_shape)) * TF_DataTypeSize(dtype),
      status));
  if (TF_GetCode(status) != TF_OK) return;
  if (ElementCount(out_shape) == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  uint32_t words = 0;
  if (!WordsPerRow(dtype, row_elements, &words, status)) return;
  BufferSlice out_slice;
  if (!SliceForTensor(out.get(), &out_slice, status)) return;

  // Later lists win where they overlap, which is what TensorFlow specifies,
  // so the scatters run in order and the stream keeps them in order.
  for (int k = 0; k < n; ++k) {
    if (index_values[k].empty()) continue;
    ScopedTensor staged;
    if (!StageIndices(ctx, index_values[k], &staged, status)) return;
    BufferSlice data_slice, index_slice;
    if (!SliceForTensor(data[k].get(), &data_slice, status)) return;
    if (!SliceForTensor(staged.get(), &index_slice, status)) return;
    if (!MoveRows(stream, /*scatter=*/true, data_slice, index_slice, out_slice,
                  static_cast<uint32_t>(index_values[k].size()), words,
                  static_cast<uint32_t>(out_shape[0]), status)) {
      return;
    }
  }
}

#define METAL_DYNAMIC_COMPUTE(NAME, BODY)                                   \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<DynamicOp*>(kernel);                             \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a data-dependent kernel has no state.");         \
    } else {                                                                \
      BODY;                                                                 \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_DYNAMIC_COMPUTE(Unique_Compute,
                      Unique_ComputeImpl(op, ctx, /*counts=*/false, status))
METAL_DYNAMIC_COMPUTE(UniqueWithCounts_Compute,
                      Unique_ComputeImpl(op, ctx, /*counts=*/true, status))
METAL_DYNAMIC_COMPUTE(DynamicPartition_Compute,
                      DynamicPartition_ComputeImpl(op, ctx, status))
METAL_DYNAMIC_COMPUTE(DynamicStitch_Compute,
                      DynamicStitch_ComputeImpl(op, ctx, status))

#undef METAL_DYNAMIC_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name, TF_DataType dtype,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &DynamicOp_Create, compute,
      &DynamicOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalDynamicKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_FLOAT, TF_INT32, TF_INT64};
  static constexpr const char* kSuffixes[] = {"Float", "Int32", "Int64"};
  for (int i = 0; i < 3; ++i) {
    Register("Unique", &Unique_Compute,
             std::string("MetalUnique") + kSuffixes[i], kDTypes[i], {});
    Register("UniqueWithCounts", &UniqueWithCounts_Compute,
             std::string("MetalUniqueWithCounts") + kSuffixes[i], kDTypes[i],
             {});
    // The partition vector decides how large each output is, so it is read on
    // the host; the data itself never leaves the device.
    Register("DynamicPartition", &DynamicPartition_Compute,
             std::string("MetalDynamicPartition") + kSuffixes[i], kDTypes[i],
             {"partitions"});
    Register("DynamicStitch", &DynamicStitch_Compute,
             std::string("MetalDynamicStitch") + kSuffixes[i], kDTypes[i],
             {"indices"});
    Register("ParallelDynamicStitch", &DynamicStitch_Compute,
             std::string("MetalParallelDynamicStitch") + kSuffixes[i],
             kDTypes[i], {"indices"});
  }
}

}  // namespace metal
}  // namespace tensorflow
