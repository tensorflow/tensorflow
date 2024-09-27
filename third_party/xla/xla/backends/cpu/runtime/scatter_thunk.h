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

#ifndef XLA_BACKENDS_CPU_RUNTIME_SCATTER_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_SCATTER_THUNK_H_

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class ScatterThunk final : public Thunk {
 public:
  using Functor = absl::AnyInvocable<void(void* result, const void** args)>;

  struct Operand {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct ScatterIndices {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct Update {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct Result {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  static absl::StatusOr<std::unique_ptr<ScatterThunk>> Create(
      Info info, std::vector<Operand> operands, ScatterIndices scatter_indices,
      std::vector<Update> updates, std::vector<Result> results,
      const ScatterDimensionNumbers& dim_numbers, std::string functor_name);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  ScatterThunk(Info info, std::vector<Operand> operands,
               ScatterIndices scatter_indices, std::vector<Update> updates,
               std::vector<Result> results,
               const ScatterDimensionNumbers& dim_numbers,
               std::string functor_name);

  std::vector<Operand> operands_;
  ScatterIndices scatter_indices_;
  std::vector<Update> updates_;
  std::vector<Result> results_;
  ScatterDimensionNumbers dim_numbers_;

  // Name of the scatter function, lazily resolved to a scatter function
  // pointer using Thunk::FunctionRegistry.
  std::string functor_name_;

  // Lazily resolved scatter functor function.
  absl::Mutex mutex_;
  std::optional<Functor> functor_ ABSL_GUARDED_BY(mutex_);
  std::atomic<Functor*> functor_ptr_;  // pointer to `functor_`
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_SCATTER_THUNK_H_
