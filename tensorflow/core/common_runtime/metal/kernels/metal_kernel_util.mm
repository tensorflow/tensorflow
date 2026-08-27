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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/core/common_runtime/metal/metal_buffer_registry.h"

namespace tensorflow {
namespace metal {

bool SliceForTensor(TF_Tensor* tensor, BufferSlice* slice, TF_Status* status) {
  void* data = TF_TensorData(tensor);
  if (data == nullptr) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: tensor has no backing data.");
    return false;
  }
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  if (!MetalBufferRegistry::Global().Lookup(data, &buffer, &offset)) {
    // Reaching this means the tensor is in host memory. Either the kernel was
    // registered with TF_KernelBuilder_HostMemory for this argument, or it ran
    // on a device this backend does not own.
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: tensor is not backed by a Metal allocation; it is "
                 "probably in host memory.");
    return false;
  }
  slice->buffer = buffer;
  slice->offset = offset;
  slice->length = TF_TensorByteSize(tensor);
  return true;
}

SP_Stream StreamForContext(TF_OpKernelContext* ctx, TF_Status* status) {
  SP_Stream stream = TF_GetStream(ctx, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  if (stream == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: op kernel context provided no stream.");
    return nullptr;
  }
  return stream;
}

id<MTLDevice> DeviceForStream(SP_Stream stream) { return stream->queue.device; }

int64_t NumElements(TF_Tensor* tensor) {
  int64_t count = 1;
  const int rank = TF_NumDims(tensor);
  for (int i = 0; i < rank; ++i) count *= TF_Dim(tensor, i);
  return count;
}

std::vector<int64_t> ShapeOf(TF_Tensor* tensor) {
  const int rank = TF_NumDims(tensor);
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for (int i = 0; i < rank; ++i) shape.push_back(TF_Dim(tensor, i));
  return shape;
}

}  // namespace metal
}  // namespace tensorflow
