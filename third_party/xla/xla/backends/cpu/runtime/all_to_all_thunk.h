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

#ifndef XLA_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/collective_thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class AllToAllThunk final : public CollectiveThunk {
 public:
  static absl::StatusOr<std::unique_ptr<AllToAllThunk>> Create(
      Info info, OpParams op_params, OpBuffers op_buffers,
      OpResources op_resources);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

 private:
  AllToAllThunk(Info info, OpParams op_params, OpBuffers op_buffers,
                OpResources op_resources);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_
