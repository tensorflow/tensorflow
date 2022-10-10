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

#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// A device-side comparator that compares buffers.
class BufferComparator {
 public:
  BufferComparator(const BufferComparator&) = delete;
  BufferComparator(BufferComparator&&) = default;

  BufferComparator(const Shape& shape, const HloModuleConfig& config);

  // Returns true if the two buffers compare equal. The definition of "equal"
  // is:
  // * All NaNs equal.
  // * All fp16 infs are treated as 65505 or -65505. Otherwise,
  //   infs and negative infs compare equal.
  // * With NaNs and infs taken care of, a and b compare equal iff:
  //     abs(a - b) / (max(abs(a), abs(b)) + 1) < tolerance
  //
  // See the implementation for the tolerance value.
  StatusOr<bool> CompareEqual(se::Stream* stream, se::DeviceMemoryBase lhs,
                              se::DeviceMemoryBase rhs) const;

 private:
  Shape shape_;
  HloModuleConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_COMPARATOR_H_
