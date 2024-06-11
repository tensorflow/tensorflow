/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_CUB_SORT_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME_CUB_SORT_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class CubSortRunnerInterface {
 public:
  virtual ~CubSortRunnerInterface() = default;
  virtual absl::Status Run(se::DeviceMemoryBase input_keys,
                           se::DeviceMemoryBase input_values,
                           se::DeviceMemoryBase output_keys,
                           se::DeviceMemoryBase output_values,
                           se::DeviceMemoryBase scratch, bool descending,
                           int64_t batch_size, se::Stream* stream) = 0;
  virtual absl::Status Run(const Thunk::ExecuteParams& params,
                           const class CubSortThunk* thunk) = 0;
  virtual absl::StatusOr<int64_t> GetScratchSize(int64_t num_items,
                                                 int64_t batch_size) = 0;

  static absl::StatusOr<std::unique_ptr<CubSortRunnerInterface>> Create(
      PrimitiveType type, std::optional<PrimitiveType> value_type);
};

class CubSortThunk : public Thunk {
 public:
  CubSortThunk(ThunkInfo thunk_info, PrimitiveType type,
               std::optional<PrimitiveType> value_type,
               absl::InlinedVector<BufferAllocation::Slice, 2> operands,
               absl::InlinedVector<BufferAllocation::Slice, 2> results,
               BufferAllocation::Slice scratch, bool descending,
               int64_t batch_size);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return runner_->Run(params, this);
  }

  BufferAllocation::Slice operand(int i) const { return operands_[i]; }
  BufferAllocation::Slice result(int i) const { return results_[i]; }
  BufferAllocation::Slice scratch() const { return scratch_; }
  bool descending() const { return descending_; }
  int64_t batch_size() const { return batch_size_; }

 private:
  std::unique_ptr<CubSortRunnerInterface> runner_;
  absl::InlinedVector<BufferAllocation::Slice, 2> operands_;
  absl::InlinedVector<BufferAllocation::Slice, 2> results_;
  BufferAllocation::Slice scratch_;
  bool descending_;
  int64_t batch_size_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_CUB_SORT_THUNK_H_
