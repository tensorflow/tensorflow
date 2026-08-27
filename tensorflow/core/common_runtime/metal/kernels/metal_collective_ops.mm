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
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// The NCCL collectives, over one device.
//
// A collective combines a tensor across the devices taking part. This backend
// has one device: every Apple silicon machine reports a single GPU, and the
// registrar refuses any device without unified memory, so a process can never
// hold two. Over one participant, a reduction of any kind is the value itself
// and a broadcast is a copy, which is what these kernels do.
//
// They are not a stub. `num_devices` is checked, and a collective that asks
// for more participants than exist is refused by name rather than answered
// with one device's data pretending to be all of them. What they are is the
// only case that can arise here, implemented exactly.

struct CollectiveOp {
  int32_t num_devices = 1;
  TF_DataType dtype = TF_FLOAT;
};

void* CollectiveOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new CollectiveOp();
  int32_t devices = 1;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_devices", &devices, status);
  if (TF_GetCode(status) == TF_OK) op->num_devices = devices;
  TF_SetStatus(status, TF_OK, "");
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &op->dtype, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_SetStatus(status, TF_OK, "");
    op->dtype = TF_FLOAT;
  }
  if (op->num_devices > 1) {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        "Metal: this backend drives a single GPU, so a collective over more "
        "than one device cannot be performed. Reduce over one device, or "
        "place the collective on the host.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void CollectiveOp_Delete(void* kernel) {
  delete static_cast<CollectiveOp*>(kernel);
}

// Over one participant every reduction is the identity, so the whole family
// reduces to a copy of the input into the output.
void Passthrough_ComputeImpl(CollectiveOp* op, TF_OpKernelContext* ctx,
                             TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  if (TF_NumOutputs(ctx) == 0) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());
  const size_t bytes = TF_TensorByteSize(input.get());
  ScopedTensor output;
  output.reset(TF_AllocateOutput(ctx, 0, TF_TensorType(input.get()),
                                 shape.data(), static_cast<int>(shape.size()),
                                 bytes, status));
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
                 "Metal: could not create a command buffer for a collective.");
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

// The send halves of the two-node forms produce nothing: with one device the
// receiving half already has the only copy there is.
void Sink_ComputeImpl(CollectiveOp* op, TF_OpKernelContext* ctx,
                      TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
}

#define METAL_COLLECTIVE_COMPUTE(NAME, IMPL)                                \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<CollectiveOp*>(kernel);                          \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a collective kernel has no state.");             \
    } else {                                                                \
      IMPL(op, ctx, status);                                                \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_COLLECTIVE_COMPUTE(Collective_Compute, Passthrough_ComputeImpl)
METAL_COLLECTIVE_COMPUTE(CollectiveSink_Compute, Sink_ComputeImpl)

#undef METAL_COLLECTIVE_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*), TF_DataType dtype,
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &CollectiveOp_Create,
                          compute, &CollectiveOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalCollectiveKernels() {
  static constexpr TF_DataType kDTypes[] = {TF_HALF, TF_FLOAT, TF_DOUBLE,
                                            TF_INT32, TF_INT64};
  static constexpr const char* kSuffixes[] = {"Half", "Float", "Double",
                                              "Int32", "Int64"};
  for (int i = 0; i < 5; ++i) {
    const std::string suffix = kSuffixes[i];
    Register("NcclAllReduce", &Collective_Compute, kDTypes[i],
             "MetalNcclAllReduce" + suffix);
    Register("NcclBroadcast", &Collective_Compute, kDTypes[i],
             "MetalNcclBroadcast" + suffix);
    Register("NcclReduce", &Collective_Compute, kDTypes[i],
             "MetalNcclReduce" + suffix);
    // The receiving halves produce the result; the sending halves have
    // nothing to send to.
    Register("_NcclBroadcastRecv", &Collective_Compute, kDTypes[i],
             "Metal_NcclBroadcastRecv" + suffix);
    Register("_NcclReduceRecv", &Collective_Compute, kDTypes[i],
             "Metal_NcclReduceRecv" + suffix);
    Register("_NcclBroadcastSend", &CollectiveSink_Compute, kDTypes[i],
             "Metal_NcclBroadcastSend" + suffix);
    Register("_NcclReduceSend", &CollectiveSink_Compute, kDTypes[i],
             "Metal_NcclReduceSend" + suffix);
  }
}

}  // namespace metal
}  // namespace tensorflow
