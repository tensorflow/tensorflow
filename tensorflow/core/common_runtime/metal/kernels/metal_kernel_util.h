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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_

// Objective-C++ only.

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {

// A tensor's storage expressed the way a Metal encoder needs it.
//
// Device tensors do not start at the beginning of an MTLBuffer: core's BFC
// allocator sub-divides one allocation into many tensors, so the offset is the
// normal case rather than an edge case. Every Metal API this backend uses to
// reach tensor memory therefore has to accept an offset, which is why
// setBuffer:offset:atIndex: and MPSMatrix's initWithBuffer:offset:descriptor:
// are preferred over interfaces that assume a buffer starts at element zero.
struct BufferSlice {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  size_t length = 0;
};

// Resolves a device tensor's storage. Fails `status` with a diagnosable
// message if the tensor's data does not lie in a live Metal allocation, which
// would mean it was placed on the host rather than the device.
bool SliceForTensor(TF_Tensor* tensor, BufferSlice* slice, TF_Status* status);

// The stream this kernel must enqueue onto. Fails `status` if the context has
// no StreamExecutor stream, which happens when the kernel was somehow placed
// on a device this backend does not own.
SP_Stream StreamForContext(TF_OpKernelContext* ctx, TF_Status* status);

// The Metal device backing `stream`.
id<MTLDevice> DeviceForStream(SP_Stream stream);

// Number of elements described by a tensor's shape.
int64_t NumElements(TF_Tensor* tensor);

// Shape as a plain vector, for shape checks and for building descriptors.
std::vector<int64_t> ShapeOf(TF_Tensor* tensor);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_KERNELS_METAL_KERNEL_UTIL_H_
