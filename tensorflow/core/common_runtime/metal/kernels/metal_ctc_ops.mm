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

// CTCLoss and CTCLossV2.
//
// The labels arrive as a sparse tensor and the sequence lengths as a vector,
// all three of them in host memory, which is where CUDA's kernel puts them
// too: they describe the shape of the work rather than take part in it. The
// forward-backward recurrence itself runs on the device, one thread per
// sequence.
//
// The two ops differ in one number. V1 treats the last class as the blank; V2
// treats class zero as the blank, because that is what cuDNN expects. Nothing
// else about them differs.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct CtcOp {
  bool blank_is_zero = false;
};

void* CtcOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new CtcOp();
  // The attributes CUDA's kernel refuses are refused here for the same
  // reason: the recurrence below implements merged repeats without
  // preprocessing, and nothing else.
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "preprocess_collapse_repeated",
                                      &flag, status);
  const bool preprocess = TF_GetCode(status) == TF_OK && flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_Bool merge = 1;
  TF_OpKernelConstruction_GetAttrBool(ctx, "ctc_merge_repeated", &merge,
                                      status);
  const bool merge_repeated = TF_GetCode(status) != TF_OK || merge != 0;
  TF_SetStatus(status, TF_OK, "");
  if (preprocess || !merge_repeated) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "Metal: CTCLoss requires preprocess_collapse_repeated to be "
                 "false and ctc_merge_repeated to be true.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void CtcOp_Delete(void* kernel) { delete static_cast<CtcOp*>(kernel); }

// Stages a host-built int32 vector where a shader can read it.
bool StageInts(TF_OpKernelContext* ctx, const std::vector<int32_t>& values,
               ScopedTensor* out, TF_Status* status) {
  int64_t dims[1] = {static_cast<int64_t>(std::max<size_t>(values.size(), 1))};
  out->reset(TF_AllocateTemp(ctx, TF_INT32, dims, 1, nullptr, status));
  if (TF_GetCode(status) != TF_OK) return false;
  void* data = TF_TensorData(out->get());
  if (data == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: a staged vector has no storage.");
    return false;
  }
  std::memset(data, 0, static_cast<size_t>(dims[0]) * sizeof(int32_t));
  if (!values.empty()) {
    std::memcpy(data, values.data(), values.size() * sizeof(int32_t));
  }
  return true;
}

void Ctc_ComputeImpl(CtcOp* op, TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor inputs, indices, values, seq_len;
  TF_GetInput(ctx, 0, inputs.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, indices.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, values.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, seq_len.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> in_shape = ShapeOf(inputs.get());
  if (in_shape.size() != 3) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: CTCLoss expects a rank-3 input.");
    return;
  }
  const int64_t max_time = in_shape[0];
  const int64_t batch = in_shape[1];
  const int64_t classes = in_shape[2];

  // The sparse labels, gathered per sequence. They are already in host memory
  // because the kernel declares them so.
  const std::vector<int64_t> index_shape = ShapeOf(indices.get());
  if (index_shape.size() != 2 || index_shape[1] != 2) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: labels_indices must have shape [n, 2].");
    return;
  }
  const int64_t entries = index_shape[0];
  const int64_t* index_data =
      static_cast<const int64_t*>(TF_TensorData(indices.get()));
  const int32_t* value_data =
      static_cast<const int32_t*>(TF_TensorData(values.get()));
  const int32_t* seq_data =
      static_cast<const int32_t*>(TF_TensorData(seq_len.get()));
  if ((index_data == nullptr && entries > 0) ||
      (value_data == nullptr && entries > 0) || seq_data == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: the CTC labels have no data.");
    return;
  }
  if (NumElements(values.get()) != entries) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: labels_values does not match labels_indices.");
    return;
  }

  std::vector<std::vector<int32_t>> per_sequence(
      static_cast<size_t>(std::max<int64_t>(batch, 0)));
  for (int64_t i = 0; i < entries; ++i) {
    const int64_t b = index_data[2 * i];
    if (b < 0 || b >= batch) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: a label index names a sequence out of range.");
      return;
    }
    per_sequence[static_cast<size_t>(b)].push_back(value_data[i]);
  }
  int64_t max_labels = 1;
  for (const auto& labels : per_sequence) {
    max_labels = std::max<int64_t>(max_labels,
                                   static_cast<int64_t>(labels.size()));
  }

  const std::vector<int64_t> loss_shape = {batch};
  ScopedTensor loss, gradient;
  loss.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, loss_shape.data(), 1,
                               static_cast<size_t>(batch) * sizeof(float),
                               status));
  if (TF_GetCode(status) != TF_OK) return;
  gradient.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, in_shape.data(), 3,
      static_cast<size_t>(ElementCount(in_shape)) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (batch == 0 || max_time == 0 || classes == 0) return;

  std::vector<int32_t> flat_labels(
      static_cast<size_t>(batch * max_labels), 0);
  std::vector<int32_t> lengths(static_cast<size_t>(batch), 0);
  std::vector<int32_t> sequence(static_cast<size_t>(batch), 0);
  for (int64_t b = 0; b < batch; ++b) {
    const auto& labels = per_sequence[static_cast<size_t>(b)];
    lengths[static_cast<size_t>(b)] = static_cast<int32_t>(labels.size());
    sequence[static_cast<size_t>(b)] = seq_data[b];
    for (size_t i = 0; i < labels.size(); ++i) {
      flat_labels[static_cast<size_t>(b * max_labels) + i] = labels[i];
    }
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  ScopedTensor staged_labels, staged_lengths, staged_sequence;
  if (!StageInts(ctx, flat_labels, &staged_labels, status)) return;
  if (!StageInts(ctx, lengths, &staged_lengths, status)) return;
  if (!StageInts(ctx, sequence, &staged_sequence, status)) return;

  // Two recurrence tables per sequence, over the extended label alphabet.
  const int64_t states = 2 * max_labels + 1;
  const std::vector<int64_t> scratch_shape = {batch * 2 * max_time * states};
  ScopedTensor scratch;
  scratch.reset(TF_AllocateTemp(ctx, TF_FLOAT, scratch_shape.data(), 1,
                                nullptr, status));
  if (TF_GetCode(status) != TF_OK) return;

  id<MTLComputePipelineState> pipeline =
      PipelineFor(DeviceForStream(stream), "tf_ctc_loss_float", status);
  if (pipeline == nil) return;

  BufferSlice in_slice, label_slice, length_slice, seq_slice, loss_slice,
      grad_slice, scratch_slice;
  if (!SliceForTensor(inputs.get(), &in_slice, status)) return;
  if (!SliceForTensor(staged_labels.get(), &label_slice, status)) return;
  if (!SliceForTensor(staged_lengths.get(), &length_slice, status)) return;
  if (!SliceForTensor(staged_sequence.get(), &seq_slice, status)) return;
  if (!SliceForTensor(loss.get(), &loss_slice, status)) return;
  if (!SliceForTensor(gradient.get(), &grad_slice, status)) return;
  if (!SliceForTensor(scratch.get(), &scratch_slice, status)) return;

  CtcParams params;
  params.batch = static_cast<uint32_t>(batch);
  params.max_time = static_cast<uint32_t>(max_time);
  params.num_classes = static_cast<uint32_t>(classes);
  params.blank = op->blank_is_zero ? 0
                                   : static_cast<uint32_t>(classes - 1);
  params.max_labels = static_cast<uint32_t>(max_labels);
  params.padding0 = 0;
  params.padding1 = 0;
  params.padding2 = 0;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for CTCLoss.");
    return;
  }
  // Time steps past a sequence's own length get no gradient, and the shader
  // never visits them, so the whole tensor starts at zero.
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:grad_slice.buffer
             range:NSMakeRange(grad_slice.offset,
                               static_cast<NSUInteger>(
                                   ElementCount(in_shape)) *
                                   sizeof(float))
             value:0];
  [zero endEncoding];

  id<MTLComputeCommandEncoder> encoder =
      [command_buffer.get() computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
  [encoder setBuffer:label_slice.buffer offset:label_slice.offset atIndex:1];
  [encoder setBuffer:length_slice.buffer offset:length_slice.offset atIndex:2];
  [encoder setBuffer:seq_slice.buffer offset:seq_slice.offset atIndex:3];
  [encoder setBuffer:loss_slice.buffer offset:loss_slice.offset atIndex:4];
  [encoder setBuffer:grad_slice.buffer offset:grad_slice.offset atIndex:5];
  [encoder setBuffer:scratch_slice.buffer
              offset:scratch_slice.offset
             atIndex:6];
  [encoder setBytes:&params length:sizeof(params) atIndex:7];
  Dispatch1D(encoder, pipeline, params.batch);
  [encoder endEncoding];
  command_buffer.Commit();
}

#define METAL_CTC_COMPUTE(NAME, BLANK_IS_ZERO)                              \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<CtcOp*>(kernel);                                 \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL, "Metal: CTCLoss has no state.");    \
    } else {                                                                \
      op->blank_is_zero = BLANK_IS_ZERO;                                    \
      Ctc_ComputeImpl(op, ctx, status);                                     \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_CTC_COMPUTE(CtcLoss_Compute, false)
METAL_CTC_COMPUTE(CtcLossV2_Compute, true)

#undef METAL_CTC_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &CtcOp_Create, compute, &CtcOp_Delete);
  // The labels and lengths describe the work rather than take part in it, so
  // they are read on the host, as CUDA's kernel reads them.
  TF_KernelBuilder_HostMemory(builder, "labels_indices");
  TF_KernelBuilder_HostMemory(builder, "labels_values");
  TF_KernelBuilder_HostMemory(builder, "sequence_length");
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

void RegisterMetalCtcKernels() {
  Register("CTCLoss", &CtcLoss_Compute, "MetalCTCLoss");
  Register("CTCLossV2", &CtcLossV2_Compute, "MetalCTCLossV2");
}

}  // namespace metal
}  // namespace tensorflow
