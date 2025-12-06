/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_H_

#include "absl/status/statusor.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// A device-side comparator that compares buffers.
class BufferComparator {
 public:
  // Maximum number of thread blocks to be used for comparator kernel
  static constexpr uint64_t kMaxNumThreadBlocksForKernel = 32768;

  BufferComparator(const BufferComparator&) = delete;
  BufferComparator(BufferComparator&&) noexcept = default;

  explicit BufferComparator(const Shape& shape, double tolerance = 0.1,
                            bool verbose = true, bool run_host_compare = true);

  // Returns true if the two buffers compare equal. The definition of "equal"
  // is:
  // * All NaNs equal.
  // * All fp16 infs are treated as 65505 or -65505. Otherwise,
  //   infs and negative infs compare equal.
  // * With NaNs and infs taken care of, a and b compare equal iff:
  //     abs(a - b) / (max(abs(a), abs(b)) + 1) < tolerance
  //
  // See the implementation for the tolerance value.
  absl::StatusOr<bool> CompareEqual(
      se::Stream* stream, const se::DeviceAddressBase& current,
      const se::DeviceAddressBase& expected) const;

 private:
  Shape shape_;
  double relative_tol_;  // relative tolerance for comparison
  bool verbose_;         // whether to print out error message on mismatch
  // enable host-side compare if device compare reports a mismatch
  bool run_host_compare_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_COMPARATOR_H_
