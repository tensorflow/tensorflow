/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_CUB_SORT_THUNK_H_
#define XLA_SERVICE_GPU_CUB_SORT_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/thunk.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class CubSortRunnerInterface {
 public:
  virtual ~CubSortRunnerInterface() = default;
  virtual Status Run(se::DeviceMemoryBase input_keys,
                     se::DeviceMemoryBase input_values,
                     se::DeviceMemoryBase output_keys,
                     se::DeviceMemoryBase output_values,
                     se::DeviceMemoryBase scratch, bool descending) = 0;
  virtual Status Run(const Thunk::ExecuteParams& params,
                     const class CubSortThunk* thunk) = 0;
  virtual StatusOr<int64_t> GetScratchSize(int64_t num_items) = 0;

  static StatusOr<std::unique_ptr<CubSortRunnerInterface>> Create(
      PrimitiveType type, std::optional<PrimitiveType> value_type);
};

class CubSortThunk : public Thunk {
 public:
  CubSortThunk(ThunkInfo thunk_info, PrimitiveType type,
               std::optional<PrimitiveType> value_type,
               std::vector<BufferAllocation::Slice> operands,
               std::vector<BufferAllocation::Slice> results,
               BufferAllocation::Slice scratch, bool descending);

  Status ExecuteOnStream(const ExecuteParams& params) override {
    return runner_->Run(params, this);
  }

  BufferAllocation::Slice operand(int i) const { return operands_[i]; }
  BufferAllocation::Slice result(int i) const { return results_[i]; }
  BufferAllocation::Slice scratch() const { return scratch_; }
  bool descending() const { return descending_; }

 private:
  std::unique_ptr<CubSortRunnerInterface> runner_;
  std::vector<BufferAllocation::Slice> operands_;
  std::vector<BufferAllocation::Slice> results_;
  BufferAllocation::Slice scratch_;
  bool descending_;
};

Status RunCubSort(PrimitiveType type, std::optional<PrimitiveType> value_type,
                  se::DeviceMemoryBase input_keys,
                  se::DeviceMemoryBase input_values,
                  se::DeviceMemoryBase output_keys,
                  se::DeviceMemoryBase output_values,
                  se::DeviceMemoryBase scratch, bool descending);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUB_SORT_THUNK_H_
