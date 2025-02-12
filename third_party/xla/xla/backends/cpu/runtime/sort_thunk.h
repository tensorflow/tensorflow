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

#ifndef XLA_BACKENDS_CPU_RUNTIME_SORT_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_SORT_THUNK_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Sorts data in the input buffers along the given dimension with a custom
// less-than comparator function.
class SortThunk final : public Thunk {
 public:
  using LessThan = absl::AnyInvocable<bool(const void** data)>;

  enum class SortDirection {
    kAscending,
    kDescending,
  };

  struct Input {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  static absl::StatusOr<std::unique_ptr<SortThunk>> Create(
      Info info, absl::Span<const Input> inputs, int64_t dimension,
      bool is_stable, LessThan less_than,
      std::optional<SortDirection> direction);

  static absl::StatusOr<std::unique_ptr<SortThunk>> Create(
      Info info, absl::Span<const Input> inputs, int64_t dimension,
      bool is_stable, std::string comparator_name,
      std::optional<SortDirection> direction);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

  std::optional<SortDirection> direction() const { return direction_; }
  int64_t dimension() const { return dimension_; }
  bool is_stable() const { return is_stable_; }
  const std::vector<Input>& inputs() const { return inputs_; }

  const std::string& comparator_name() const { return comparator_name_; }

  bool has_less_than() const { return less_than_.ok(); }

 private:
  SortThunk(Info info, absl::Span<const Input> inputs, int64_t dimension,
            bool is_stable, LessThan less_than,
            std::optional<SortDirection> direction);

  SortThunk(Info info, absl::Span<const Input> inputs, int64_t dimension,
            bool is_stable, std::string comparator_name,
            std::optional<SortDirection> direction);

  std::vector<Input> inputs_;
  int64_t dimension_;
  bool is_stable_;
  std::optional<SortDirection> direction_;

  // Name of the comparator function, lazily resolved to a comparator function
  // pointer using Thunk::FunctionRegistry.
  std::string comparator_name_;

  // Lazily resolved LessThan comparator function.
  absl::once_flag less_than_init_flag_;
  absl::StatusOr<LessThan> less_than_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_SORT_THUNK_H_
