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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ALL_REDUCE_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_ALL_REDUCE_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

class AllReduceThunk final : public CollectiveThunk {
 public:
  static absl::StatusOr<std::unique_ptr<AllReduceThunk>> Create(
      Info info, ReductionKind reduction_kind, OpParams op_params,
      OpBuffers op_buffers, OpResources op_resources, bool single_replica);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  ReductionKind reduction_kind() const { return reduction_kind_; }
  bool single_replica() const { return single_replica_; }

 private:
  AllReduceThunk(Info info, ReductionKind reduction_kind, OpParams op_params,
                 OpBuffers op_buffers, OpResources op_resources,
                 bool single_replica);

  ReductionKind reduction_kind_;
  bool single_replica_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ALL_REDUCE_THUNK_H_
