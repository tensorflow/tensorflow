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

// The sparse segment reductions and their gradients.
//
// These are what an embedding lookup reduces with: gather the rows named by
// `indices` and sum each one into the segment `segment_ids` assigns it to.
// Many rows land in the same segment, so the accumulation is atomic, and the
// mean and square-root forms are the same sum divided afterwards by a count
// gathered the same way.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

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

// 0 sum, 1 mean, 2 square root of the count.
enum class Mode { kSum = 0, kMean = 1, kSqrtN = 2 };

struct SegmentOp {
  Mode mode = Mode::kSum;
  // The gradient forms; V2 emits a sparse result instead of a dense one.
  bool gradient = false;
  bool sparse_gradient = false;
  // The forward form that is told how many segments there are.
  bool num_segments_input = false;
};

void* SegmentOp_Create(TF_OpKernelConstruction* ctx) {
  // Every distinction is fixed by which op is being built, so the kernel's
  // state is set by the registration rather than read from attributes.
  return new SegmentOp();
}

void SegmentOp_Delete(void* kernel) { delete static_cast<SegmentOp*>(kernel); }

const char* TypeSuffix(TF_DataType index, TF_DataType segment) {
  if (index == TF_INT32) {
    return segment == TF_INT32 ? "_i32_i32" : "_i32_i64";
  }
  return segment == TF_INT32 ? "_i64_i32" : "_i64_i64";
}

// Reads an int32 or int64 scalar that lives in host memory.
bool ReadHostScalar(TF_Tensor* t, int64_t* out, TF_Status* status) {
  const void* data = TF_TensorData(t);
  if (data == nullptr || TF_TensorElementCount(t) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a segment parameter has no data.");
    return false;
  }
  *out = TF_TensorType(t) == TF_INT64
             ? *static_cast<const int64_t*>(data)
             : *static_cast<const int32_t*>(data);
  return true;
}

// Reads a whole index vector on the host. Only the forms whose output length
// depends on the values need this.
bool ReadIndexVector(TF_Tensor* t, std::vector<int64_t>* out,
                     TF_Status* status) {
  const int64_t count = TF_TensorElementCount(t);
  const void* data = TF_TensorData(t);
  if (data == nullptr && count > 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: an index vector has no storage.");
    return false;
  }
  out->resize(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    (*out)[static_cast<size_t>(i)] =
        TF_TensorType(t) == TF_INT64
            ? static_cast<const int64_t*>(data)[i]
            : static_cast<const int32_t*>(data)[i];
  }
  return true;
}

bool StageInts(TF_OpKernelContext* ctx, const std::vector<int32_t>& values,
               ScopedTensor* out, TF_Status* status) {
  int64_t dims[1] = {static_cast<int64_t>(std::max<size_t>(values.size(), 1))};
  out->reset(TF_AllocateTemp(ctx, TF_INT32, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out->get());
  if (data == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Metal: a staged vector has no storage.");
    return false;
  }
  std::memset(data, 0, static_cast<size_t>(dims[0]) * sizeof(int32_t));
  if (!values.empty()) {
    std::memcpy(data, values.data(), values.size() * sizeof(int32_t));
  }
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

bool Dispatch(SP_Stream stream, const char* function,
              const std::vector<BufferSlice>& buffers,
              const SegmentParams& params, uint32_t threads,
              TF_Status* status) {
  if (threads == 0) return true;
  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), function, status);
  if (pipeline == nil) return false;
  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a segment "
                 "reduction.");
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

/*** FORWARD ***/

void SegmentForward_ComputeImpl(SegmentOp* op, TF_OpKernelContext* ctx,
                                TF_Status* status) {
  ScopedTensor data, indices, segment_ids;
  TF_GetInput(ctx, 0, data.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, segment_ids.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> data_shape = ShapeOf(data.get());
  if (data_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a sparse segment reduction needs a rank of at least "
                 "one.");
    return;
  }
  const int64_t num_indices = NumElements(indices.get());
  if (NumElements(segment_ids.get()) != num_indices) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: indices and segment_ids must have the same length.");
    return;
  }
  int64_t inner = 1;
  for (size_t i = 1; i < data_shape.size(); ++i) inner *= data_shape[i];

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  int64_t num_segments = 0;
  if (op->num_segments_input) {
    ScopedTensor given;
    TF_GetInput(ctx, 3, given.address(), status);
    if (TF_GetCode(status) != TF_OK) return;
    if (!ReadHostScalar(given.get(), &num_segments, status)) return;
  } else {
    // Without it, the number of segments is one past the largest identifier,
    // which cannot be known before the identifiers exist.
    WaitForStream(stream);
    std::vector<int64_t> ids;
    if (!ReadIndexVector(segment_ids.get(), &ids, status)) return;
    for (int64_t id : ids) num_segments = std::max(num_segments, id + 1);
  }
  if (num_segments < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the number of segments must not be negative.");
    return;
  }

  std::vector<int64_t> out_shape = data_shape;
  out_shape[0] = num_segments;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(ElementCount(out_shape)) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, output.get(), status)) return;
  if (num_segments == 0 || inner == 0) return;

  // One count per segment, for the two normalised forms and for nothing else.
  const std::vector<int64_t> count_shape = {num_segments};
  ScopedTensor counts;
  counts.reset(
      TF_AllocateTemp(ctx, TF_INT32, count_shape.data(), 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, counts.get(), status)) return;

  SegmentParams params;
  params.num_indices = static_cast<uint32_t>(num_indices);
  params.inner = static_cast<uint32_t>(inner);
  params.num_segments = static_cast<uint32_t>(num_segments);
  params.data_rows = static_cast<uint32_t>(data_shape[0]);
  params.mode = static_cast<uint32_t>(op->mode);
  params.count = static_cast<uint32_t>(num_indices * inner);
  params.padding0 = 0;
  params.padding1 = 0;

  std::vector<BufferSlice> buffers(5);
  if (!SliceForTensor(data.get(), &buffers[0], status)) return;
  if (!SliceForTensor(indices.get(), &buffers[1], status)) return;
  if (!SliceForTensor(segment_ids.get(), &buffers[2], status)) return;
  if (!SliceForTensor(output.get(), &buffers[3], status)) return;
  if (!SliceForTensor(counts.get(), &buffers[4], status)) return;

  const std::string forward =
      std::string("tf_sparse_segment_forward") +
      TypeSuffix(TF_TensorType(indices.get()), TF_TensorType(segment_ids.get()));
  if (!Dispatch(stream, forward.c_str(), buffers, params, params.count,
                status)) {
    return;
  }
  if (op->mode == Mode::kSum) return;

  SegmentParams scale = params;
  scale.count = static_cast<uint32_t>(num_segments * inner);
  std::vector<BufferSlice> scale_buffers = {buffers[3], buffers[4]};
  Dispatch(stream, "tf_sparse_segment_normalise_float", scale_buffers, scale,
           scale.count, status);
}

/*** GRADIENTS ***/

void SegmentGrad_ComputeImpl(SegmentOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor grad, indices, segment_ids, dim0;
  TF_GetInput(ctx, 0, grad.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, segment_ids.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, dim0.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> grad_shape = ShapeOf(grad.get());
  if (grad_shape.empty()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a segment gradient needs a rank of at least one.");
    return;
  }
  int64_t inner = 1;
  for (size_t i = 1; i < grad_shape.size(); ++i) inner *= grad_shape[i];
  const int64_t num_indices = NumElements(indices.get());
  const int64_t num_segments = grad_shape[0];
  int64_t dense_rows = 0;
  if (!ReadHostScalar(dim0.get(), &dense_rows, status)) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  // The sparse form emits one row per distinct index, in increasing order,
  // together with the indices themselves. Which indices those are depends on
  // the values, so they are read here.
  std::vector<int64_t> unique;
  ScopedTensor remapped;
  int64_t out_rows = dense_rows;
  if (op->sparse_gradient) {
    WaitForStream(stream);
    std::vector<int64_t> raw;
    if (!ReadIndexVector(indices.get(), &raw, status)) return;
    unique = raw;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    out_rows = static_cast<int64_t>(unique.size());
    std::vector<int32_t> positions(raw.size(), 0);
    for (size_t i = 0; i < raw.size(); ++i) {
      const auto it =
          std::lower_bound(unique.begin(), unique.end(), raw[i]);
      positions[i] = static_cast<int32_t>(it - unique.begin());
    }
    if (!StageInts(ctx, positions, &remapped, status)) return;
  }

  std::vector<int64_t> out_shape = grad_shape;
  out_shape[0] = out_rows;
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, out_shape.data(), static_cast<int>(out_shape.size()),
      static_cast<size_t>(ElementCount(out_shape)) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, output.get(), status)) return;

  if (op->sparse_gradient) {
    const std::vector<int64_t> index_shape = {out_rows};
    ScopedTensor sorted;
    const TF_DataType index_dtype = TF_TensorType(indices.get());
    sorted.reset(TF_AllocateOutput(
        ctx, 1, index_dtype, index_shape.data(), 1,
        static_cast<size_t>(out_rows) * TF_DataTypeSize(index_dtype), status));
    if (TF_GetCode(status) != TF_OK) return;
    void* data = TF_TensorData(sorted.get());
    if (data != nullptr) {
      for (int64_t i = 0; i < out_rows; ++i) {
        if (index_dtype == TF_INT64) {
          static_cast<int64_t*>(data)[i] = unique[static_cast<size_t>(i)];
        } else {
          static_cast<int32_t*>(data)[i] =
              static_cast<int32_t>(unique[static_cast<size_t>(i)]);
        }
      }
    }
  }
  if (num_indices == 0 || inner == 0 || out_rows == 0) return;

  const std::vector<int64_t> count_shape = {std::max<int64_t>(num_segments, 1)};
  ScopedTensor counts;
  counts.reset(
      TF_AllocateTemp(ctx, TF_INT32, count_shape.data(), 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (!ZeroTensor(stream, counts.get(), status)) return;

  SegmentParams params;
  params.num_indices = static_cast<uint32_t>(num_indices);
  params.inner = static_cast<uint32_t>(inner);
  params.num_segments = static_cast<uint32_t>(num_segments);
  params.data_rows = static_cast<uint32_t>(out_rows);
  params.mode = static_cast<uint32_t>(op->mode);
  params.count = static_cast<uint32_t>(num_indices * inner);
  params.padding0 = 0;
  params.padding1 = 0;

  BufferSlice segment_slice, counts_slice;
  if (!SliceForTensor(segment_ids.get(), &segment_slice, status)) return;
  if (!SliceForTensor(counts.get(), &counts_slice, status)) return;
  const TF_DataType segment_dtype = TF_TensorType(segment_ids.get());
  if (op->mode != Mode::kSum) {
    // The divisor is the same one the forward pass used, so it is counted
    // again rather than carried between the two ops.
    const std::string counter = std::string("tf_sparse_segment_counts") +
                                (segment_dtype == TF_INT64 ? "_i64" : "_i32");
    std::vector<BufferSlice> count_buffers = {segment_slice, counts_slice};
    if (!Dispatch(stream, counter.c_str(), count_buffers, params,
                  params.num_indices, status)) {
      return;
    }
  }

  BufferSlice index_slice;
  TF_DataType index_dtype = TF_TensorType(indices.get());
  if (op->sparse_gradient) {
    if (!SliceForTensor(remapped.get(), &index_slice, status)) return;
    index_dtype = TF_INT32;
  } else {
    if (!SliceForTensor(indices.get(), &index_slice, status)) return;
  }

  std::vector<BufferSlice> buffers(5);
  if (!SliceForTensor(grad.get(), &buffers[0], status)) return;
  buffers[1] = index_slice;
  buffers[2] = segment_slice;
  buffers[3] = counts_slice;
  if (!SliceForTensor(output.get(), &buffers[4], status)) return;

  const std::string function = std::string("tf_sparse_segment_grad") +
                               TypeSuffix(index_dtype, segment_dtype);
  Dispatch(stream, function.c_str(), buffers, params, params.count, status);
}

#define METAL_SEGMENT_COMPUTE(NAME, MODE, GRADIENT, SPARSE, NUM_SEGMENTS)   \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<SegmentOp*>(kernel);                             \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a segment kernel has no state.");                \
    } else {                                                                \
      op->mode = MODE;                                                      \
      op->gradient = GRADIENT;                                              \
      op->sparse_gradient = SPARSE;                                         \
      op->num_segments_input = NUM_SEGMENTS;                                \
      if (GRADIENT) {                                                       \
        SegmentGrad_ComputeImpl(op, ctx, status);                           \
      } else {                                                              \
        SegmentForward_ComputeImpl(op, ctx, status);                        \
      }                                                                     \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_SEGMENT_COMPUTE(SumForward_Compute, Mode::kSum, false, false, false)
METAL_SEGMENT_COMPUTE(MeanForward_Compute, Mode::kMean, false, false, false)
METAL_SEGMENT_COMPUTE(SqrtNForward_Compute, Mode::kSqrtN, false, false, false)
METAL_SEGMENT_COMPUTE(SumSegments_Compute, Mode::kSum, false, false, true)
METAL_SEGMENT_COMPUTE(MeanSegments_Compute, Mode::kMean, false, false, true)
METAL_SEGMENT_COMPUTE(SqrtNSegments_Compute, Mode::kSqrtN, false, false, true)
METAL_SEGMENT_COMPUTE(SumGrad_Compute, Mode::kSum, true, false, false)
METAL_SEGMENT_COMPUTE(MeanGrad_Compute, Mode::kMean, true, false, false)
METAL_SEGMENT_COMPUTE(SqrtNGrad_Compute, Mode::kSqrtN, true, false, false)
METAL_SEGMENT_COMPUTE(SumGradV2_Compute, Mode::kSum, true, true, false)
METAL_SEGMENT_COMPUTE(MeanGradV2_Compute, Mode::kMean, true, true, false)
METAL_SEGMENT_COMPUTE(SqrtNGradV2_Compute, Mode::kSqrtN, true, true, false)

#undef METAL_SEGMENT_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType index,
              TF_DataType segment, const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &SegmentOp_Create, compute,
      &SegmentOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "Tidx", index, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_KernelBuilder_TypeConstraint(builder, "Tsegmentids", segment, status);
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

void RegisterMetalSparseSegmentKernels() {
  static constexpr TF_DataType kTypes[] = {TF_INT32, TF_INT64};
  static constexpr const char* kNames[] = {"Int32", "Int64"};
  struct Entry {
    const char* op;
    void (*compute)(void*, TF_OpKernelContext*);
    const char* name;
    bool num_segments;
    bool gradient;
  };
  static const Entry kEntries[] = {
      {"SparseSegmentSum", &SumForward_Compute, "MetalSparseSegmentSum", false,
       false},
      {"SparseSegmentMean", &MeanForward_Compute, "MetalSparseSegmentMean",
       false, false},
      {"SparseSegmentSqrtN", &SqrtNForward_Compute, "MetalSparseSegmentSqrtN",
       false, false},
      {"SparseSegmentSumWithNumSegments", &SumSegments_Compute,
       "MetalSparseSegmentSumWithNumSegments", true, false},
      {"SparseSegmentMeanWithNumSegments", &MeanSegments_Compute,
       "MetalSparseSegmentMeanWithNumSegments", true, false},
      {"SparseSegmentSqrtNWithNumSegments", &SqrtNSegments_Compute,
       "MetalSparseSegmentSqrtNWithNumSegments", true, false},
      {"SparseSegmentSumGrad", &SumGrad_Compute, "MetalSparseSegmentSumGrad",
       false, true},
      {"SparseSegmentMeanGrad", &MeanGrad_Compute,
       "MetalSparseSegmentMeanGrad", false, true},
      {"SparseSegmentSqrtNGrad", &SqrtNGrad_Compute,
       "MetalSparseSegmentSqrtNGrad", false, true},
      {"SparseSegmentSumGradV2", &SumGradV2_Compute,
       "MetalSparseSegmentSumGradV2", false, true},
      {"SparseSegmentMeanGradV2", &MeanGradV2_Compute,
       "MetalSparseSegmentMeanGradV2", false, true},
      {"SparseSegmentSqrtNGradV2", &SqrtNGradV2_Compute,
       "MetalSparseSegmentSqrtNGradV2", false, true},
  };
  for (const Entry& entry : kEntries) {
    // The segment count and the dense row count size the output, so they are
    // read on the host; nothing else leaves the device.
    std::vector<const char*> host;
    if (entry.num_segments) host.push_back("num_segments");
    if (entry.gradient) host.push_back("dense_output_dim0");
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        Register(entry.op, entry.compute, kTypes[i], kTypes[j],
                 std::string(entry.name) + kNames[i] + kNames[j], host);
      }
    }
  }
}

}  // namespace metal
}  // namespace tensorflow
