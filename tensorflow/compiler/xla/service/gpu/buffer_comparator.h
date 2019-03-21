/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_COMPARATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_COMPARATOR_H_

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A device-side comparator that compares buffers.
class BufferComparator {
 public:
  BufferComparator(const BufferComparator&) = delete;
  BufferComparator(BufferComparator&&) = default;

  static StatusOr<BufferComparator> Create(const Shape& buffer_shape,
                                           se::StreamExecutor* stream_exec,
                                           Compiler* compiler);

  // Returns true if the two buffers compare equal. The definition of "equal"
  // is:
  // * All NaNs equal.
  // * All fp16 infs are treated as 65505 or -65505. Otherwise,
  //   infs and negative infs compare equal.
  // * With NaNs and infs taken care of, a and b compare equal iff:
  //     abs(a - b) / (max(abs(a), abs(b)) + 1) < tolerance
  //
  // See the implementation for the tolerance value.
  StatusOr<bool> CompareEqual(se::Stream* stream,
                              DeviceMemoryAllocator* allocator,
                              se::DeviceMemoryBase lhs,
                              se::DeviceMemoryBase rhs);

 private:
  BufferComparator(const Shape& shape, std::unique_ptr<Executable> exec)
      : shape_(shape), comparator_exec_(std::move(exec)) {}

  StatusOr<bool> CompareEqualImpl(se::Stream* stream,
                                  DeviceMemoryAllocator* allocator,
                                  se::DeviceMemoryBase lhs,
                                  se::DeviceMemoryBase rhs);

  Shape shape_;
  std::unique_ptr<Executable> comparator_exec_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_COMPARATOR_H_
