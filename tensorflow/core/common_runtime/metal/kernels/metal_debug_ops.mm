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
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {
namespace metal {
namespace {

// DebugNumericSummaryV2 and _TensorToHashBucketFast.
//
// The summary is a reduction and stays on the device. Its leading slots
// describe the tensor rather than its contents, and the host is the only one
// that knows them, so they are written by a first pass and the counts
// accumulate in a second. Two encoders in one command buffer, which Metal
// orders, so the counts never land before the slots they sit beside.

int64_t ElementCount(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

struct DebugOp {
  int32_t mode = 2;
  int64_t tensor_id = -1;
};

void* DebugOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DebugOp();
  int32_t mode = 2;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "tensor_debug_mode", &mode,
                                       status);
  if (TF_GetCode(status) == TF_OK) op->mode = mode;
  TF_SetStatus(status, TF_OK, "");
  int64_t id = -1;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "tensor_id", &id, status);
  if (TF_GetCode(status) == TF_OK) op->tensor_id = id;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void DebugOp_Delete(void* kernel) { delete static_cast<DebugOp*>(kernel); }

// The length of the summary and how many of its leading slots describe the
// tensor rather than its contents.
bool ModeLayout(int mode, int64_t* length, int64_t* prefix,
                const char** function, TF_Status* status) {
  switch (mode) {
    case 2:  // Whether anything is not finite.
      *length = 2;
      *prefix = 1;
      *function = "tf_debug_curt_health_float";
      return true;
    case 3:  // Counts of the three kinds of non-finite value.
      *length = 5;
      *prefix = 2;
      *function = "tf_debug_concise_health_float";
      return true;
    case 4:  // Those counts plus the signs of everything finite.
      *length = 11;
      *prefix = 5;
      *function = "tf_debug_full_health_float";
      return true;
    case 5:  // The shape alone, which needs no pass over the data.
      *length = 10;
      *prefix = 10;
      *function = nullptr;
      return true;
    case 8:  // The three offending values, for further reduction.
      *length = 3;
      *prefix = 0;
      *function = "tf_debug_three_slots_float";
      return true;
    default:
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "Metal: this numeric summary mode is not implemented.");
      return false;
  }
}

void DebugSummary_ComputeImpl(DebugOp* op, TF_OpKernelContext* ctx,
                              TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);

  int64_t length = 0, prefix = 0;
  const char* function = nullptr;
  if (!ModeLayout(op->mode, &length, &prefix, &function, status)) return;

  const std::vector<int64_t> out_shape = {length};
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_FLOAT, out_shape.data(), 1,
                                 static_cast<size_t>(length) * sizeof(float),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;

  DebugParams params;
  params.count = static_cast<uint32_t>(count);
  params.prefix_count = static_cast<uint32_t>(prefix);
  params.padding0 = 0;
  params.padding1 = 0;
  for (int i = 0; i < 10; ++i) params.prefix[i] = 0.0f;
  const float tensor_id = static_cast<float>(op->tensor_id);
  const float dtype = static_cast<float>(TF_TensorType(input.get()));
  const float rank = static_cast<float>(shape.size());
  switch (op->mode) {
    case 2:
      params.prefix[0] = tensor_id;
      break;
    case 3:
      params.prefix[0] = tensor_id;
      params.prefix[1] = static_cast<float>(count);
      break;
    case 4:
      params.prefix[0] = tensor_id;
      // The second slot is the device identifier, which TensorFlow has never
      // filled in and reports as minus one.
      params.prefix[1] = -1.0f;
      params.prefix[2] = dtype;
      params.prefix[3] = rank;
      params.prefix[4] = static_cast<float>(count);
      break;
    case 5: {
      params.prefix[0] = tensor_id;
      params.prefix[1] = dtype;
      params.prefix[2] = rank;
      params.prefix[3] = static_cast<float>(count);
      // Six dimensions are reported; a deeper tensor keeps its leading six.
      for (size_t i = 0; i < shape.size() && i < 6; ++i) {
        params.prefix[4 + i] = static_cast<float>(shape[i]);
      }
      break;
    }
    default:
      break;
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  id<MTLDevice> device = DeviceForStream(stream);
  BufferSlice in_slice, out_slice;
  if (count > 0 && !SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  id<MTLComputePipelineState> prefix_pipeline =
      PipelineFor(device, "tf_debug_prefix_float", status);
  if (prefix_pipeline == nil) return;
  id<MTLComputePipelineState> count_pipeline = nil;
  if (function != nullptr && count > 0) {
    count_pipeline = PipelineFor(device, function, status);
    if (count_pipeline == nil) return;
  }

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for a numeric "
                 "summary.");
    return;
  }
  id<MTLBlitCommandEncoder> zero = [command_buffer.get() blitCommandEncoder];
  [zero fillBuffer:out_slice.buffer
             range:NSMakeRange(out_slice.offset,
                               static_cast<NSUInteger>(length) * sizeof(float))
             value:0];
  [zero endEncoding];

  if (prefix > 0) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:prefix_pipeline];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:0];
    [encoder setBytes:&params length:sizeof(params) atIndex:1];
    Dispatch1D(encoder, prefix_pipeline, params.prefix_count);
    [encoder endEncoding];
  }
  if (count_pipeline != nil) {
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer.get() computeCommandEncoder];
    [encoder setComputePipelineState:count_pipeline];
    [encoder setBuffer:in_slice.buffer offset:in_slice.offset atIndex:0];
    [encoder setBuffer:out_slice.buffer offset:out_slice.offset atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    Dispatch1D(encoder, count_pipeline, params.count);
    [encoder endEncoding];
  }
  command_buffer.Commit();
}

/*** TENSOR TO HASH BUCKET ***/

// Renders each integer as its decimal string and hashes it. The hash has to
// match the host's bucket for bucket, or a graph that moves this op between
// devices would sort its keys differently; the surest way to match a
// fingerprint is to call the same function, so this waits for the stream and
// hashes on the host rather than reimplementing the fingerprint in a shader.
void HashBucket_ComputeImpl(DebugOp* op, TF_OpKernelContext* ctx,
                            TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  const std::vector<int64_t> shape = ShapeOf(input.get());
  const int64_t count = ElementCount(shape);
  const TF_DataType dtype = TF_TensorType(input.get());

  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_INT64, shape.data(),
                                 static_cast<int>(shape.size()),
                                 static_cast<size_t>(count) * sizeof(int64_t),
                                 status));
  if (TF_GetCode(status) != TF_OK) return;
  if (count == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  {
    uint64_t target = 0;
    {
      absl::MutexLock lock(&stream->mu);
      target = stream->last_enqueued;
    }
    if (target > 0) {
      [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
    }
  }

  const void* data = TF_TensorData(input.get());
  int64_t* out = static_cast<int64_t*>(TF_TensorData(output.get()));
  if (data == nullptr || out == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: the hash bucket inputs have no storage.");
    return;
  }
  const int64_t buckets = op->tensor_id;  // num_buckets, read at construction
  if (buckets <= 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: num_buckets must be positive.");
    return;
  }
  for (int64_t i = 0; i < count; ++i) {
    int64_t value = 0;
    switch (dtype) {
      case TF_INT8:
        value = static_cast<const int8_t*>(data)[i];
        break;
      case TF_INT16:
        value = static_cast<const int16_t*>(data)[i];
        break;
      case TF_INT32:
        value = static_cast<const int32_t*>(data)[i];
        break;
      default:
        value = static_cast<const int64_t*>(data)[i];
        break;
    }
    const std::string rendered = std::to_string(value);
    out[i] = static_cast<int64_t>(Fingerprint64(rendered) % buckets);
  }
}

#define METAL_DEBUG_COMPUTE(NAME, IMPL)                                     \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<DebugOp*>(kernel);                               \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL, "Metal: kernel has no state.");     \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_DEBUG_COMPUTE(DebugSummary_Compute, DebugSummary_ComputeImpl)
METAL_DEBUG_COMPUTE(HashBucket_Compute, HashBucket_ComputeImpl)

#undef METAL_DEBUG_COMPUTE

// The hash op names its bucket count `num_buckets`; the summary names its
// identifier `tensor_id`. One state object serves both, so the constructor
// reads whichever is present.
void* HashOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new DebugOp();
  int64_t buckets = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "num_buckets", &buckets, status);
  if (TF_GetCode(status) == TF_OK) op->tensor_id = buckets;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void Register(const char* op_name, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*),
              const char* type_attr, TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, create, compute, &DebugOp_Delete);
  if (type_attr != nullptr) {
    TF_KernelBuilder_TypeConstraint(builder, type_attr, dtype, status);
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

void RegisterMetalDebugKernels() {
  Register("DebugNumericSummaryV2", &DebugOp_Create, &DebugSummary_Compute,
           "output_dtype", TF_FLOAT, "MetalDebugNumericSummaryV2Float");
  static constexpr TF_DataType kInts[] = {TF_INT8, TF_INT16, TF_INT32,
                                          TF_INT64};
  static constexpr const char* kNames[] = {"Int8", "Int16", "Int32", "Int64"};
  for (int i = 0; i < 4; ++i) {
    Register("_TensorToHashBucketFast", &HashOp_Create, &HashBucket_Compute,
             "T", kInts[i],
             std::string("Metal_TensorToHashBucketFast") + kNames[i]);
  }
}

}  // namespace metal
}  // namespace tensorflow
