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

// Identity is worth having early even though it computes nothing: TensorFlow
// inserts Identity nodes throughout real graphs, for control dependencies,
// for variable reads and around function boundaries. Without a Metal kernel
// for it, every one of those nodes would be placed on the host and drag a
// device-to-host and host-to-device round trip with it.

void IdentityOp_ComputeImpl(TF_OpKernelContext* ctx, TF_Status* status) {
  ScopedTensor input;
  TF_GetInput(ctx, 0, input.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> shape = ShapeOf(input.get());

  // Ask core to alias the input buffer as the output. When the input is not
  // shared with anything else this succeeds and Identity costs nothing at all,
  // which is the whole point of having the kernel.
  int forwarded_input = -1;
  const int candidate = 0;
  ScopedTensor output;
  output.reset(TF_ForwardInputOrAllocateOutput(
      ctx, &candidate, 1, 0, shape.data(), static_cast<int>(shape.size()),
      &forwarded_input, status));
  if (TF_GetCode(status) != TF_OK) return;
  if (forwarded_input == 0) return;  // Aliased; nothing to copy.

  const size_t bytes = TF_TensorByteSize(input.get());
  if (bytes == 0) return;

  // The input was shared, so core allocated a separate output and the bytes
  // have to be moved. A blit keeps the copy on the GPU and in stream order.
  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;

  BufferSlice in_slice;
  BufferSlice out_slice;
  if (!SliceForTensor(input.get(), &in_slice, status)) return;
  if (!SliceForTensor(output.get(), &out_slice, status)) return;

  OrderedCommandBuffer command_buffer(stream);
  if (!command_buffer.ok()) {
    TF_SetStatus(status, TF_RESOURCE_EXHAUSTED,
                 "Metal: could not create a command buffer for Identity.");
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

void IdentityOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  IdentityOp_ComputeImpl(ctx, status);
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void RegisterIdentity(const char* op_name, TF_DataType dtype,
                      const std::string& kernel_name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, /*create_func=*/nullptr, &IdentityOp_Compute,
      /*delete_func=*/nullptr);
  TF_KernelBuilder_TypeConstraint(builder, "T", dtype, status);
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

void RegisterMetalIdentityKernels() {
  // Identity carries no state, so unlike the arithmetic kernels it needs no
  // create callback and Compute ignores the kernel pointer.
  struct DTypeEntry {
    TF_DataType dtype;
    const char* suffix;
  };
  // int32 only, and that is the whole list on purpose.
  //
  // TensorFlow registers Identity for DEVICE_GPU itself, outside any CUDA
  // guard, for every number type except int32 plus bool, so those apply to
  // this device already. Registering them again produced two registrations
  // TensorFlow cannot choose between, and it refuses to run an op whose
  // registrations tie: Identity, which is in nearly every graph, could not
  // execute on the GPU at all.
  //
  // int32 is the exception in both directions. TensorFlow's number-type lists
  // leave it out, and its host-memory int32 kernel sits behind
  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM, so in this build nothing else
  // registers it.
  static constexpr DTypeEntry kDTypes[] = {
      {TF_INT32, "Int32"},
  };
  for (const DTypeEntry& entry : kDTypes) {
    RegisterIdentity("Identity", entry.dtype,
                     std::string("MetalIdentity") + entry.suffix);
  }
}

}  // namespace metal
}  // namespace tensorflow
