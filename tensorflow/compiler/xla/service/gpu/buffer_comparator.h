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
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A fp16 comparator that internally keeps a reference buffer, and compares it
// against other test buffers.
class F16BufferComparator {
 public:
  F16BufferComparator(const F16BufferComparator&) = delete;
  F16BufferComparator(F16BufferComparator&&) = default;

  // Creates a new comparator. It internally allocates a buffer initialized by
  // ref_buffer.
  static StatusOr<F16BufferComparator> Create(
      se::DeviceMemory<Eigen::half> ref_buffer, Compiler* compiler,
      DeviceMemoryAllocator* allocator, se::Stream* stream);

  // Returns true if the internally allocated buffer "compares equal" to
  // test_buffer. The definition of "equal" is:
  // * All NaNs equal.
  // * All infs are treated as 65505 or -65505, so that this checker is tolerant
  //   to fp16 overflows.
  // * With NaNs and infs taken care of, a and b compare equal iff:
  //     abs(a - b) / (max(abs(a), abs(b)) + 1) < tolerance
  //
  // See the implementation for the tolerance value.
  StatusOr<bool> CompareEqual(se::DeviceMemory<Eigen::half> test_buffer);

 private:
  F16BufferComparator(se::Stream* stream, DeviceMemoryAllocator* allocator,
                      std::unique_ptr<Executable> exec,
                      ScopedShapedBuffer ref_buffer)
      : stream_(stream),
        allocator_(allocator),
        exec_(std::move(exec)),
        ref_buffer_(std::move(ref_buffer)) {}

  StatusOr<bool> CompareEqualImpl(se::DeviceMemory<Eigen::half> test_buffer);

  se::Stream* stream_;
  DeviceMemoryAllocator* allocator_;
  std::unique_ptr<Executable> exec_;
  ScopedShapedBuffer ref_buffer_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_COMPARATOR_H_
