/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class CollectivePermuteThunk final : public CollectiveThunk {
 public:
  using SourceTargetPair = std::pair<int64_t, int64_t>;

  static absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>> Create(
      Info info, OpParams op_params, OpBuffers op_buffers,
      OpResources op_resources,
      absl::Span<const SourceTargetPair> source_target_pairs);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  const std::vector<SourceTargetPair>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  CollectivePermuteThunk(
      Info info, OpParams op_params, OpBuffers op_buffers,
      OpResources op_resources,
      absl::Span<const SourceTargetPair> source_target_pairs);

  std::vector<SourceTargetPair> source_target_pairs_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
