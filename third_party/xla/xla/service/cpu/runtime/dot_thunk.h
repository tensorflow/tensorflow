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

#ifndef XLA_SERVICE_CPU_RUNTIME_DOT_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_DOT_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class DotThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<DotThunk>> Create(
      Info info, DotDimensionNumbers dot_dimensions,
      BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
      BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
      BufferAllocation::Slice out_buffer, Shape out_shape);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final {
    return {BufferUse::Read(lhs_buffer_), BufferUse::Read(rhs_buffer_),
            BufferUse::Write(out_buffer_)};
  }

 private:
  DotThunk(Info info, DotDimensionNumbers dot_dimensions,
           BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
           BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
           BufferAllocation::Slice out_buffer, Shape out_shape,
           int64_t batch_size, Shape lhs_matmul_shape, Shape rhs_matmul_shape,
           Shape out_matmul_shape);

  DotDimensionNumbers dot_dimensions_;

  BufferAllocation::Slice lhs_buffer_;
  Shape lhs_shape_;

  BufferAllocation::Slice rhs_buffer_;
  Shape rhs_shape_;

  BufferAllocation::Slice out_buffer_;
  Shape out_shape_;

  // Product of batch dimensions.
  int64_t batch_size_;

  // Shapes of the non-batch matrix-multiplication for the dot operation
  Shape lhs_matmul_shape_;
  Shape rhs_matmul_shape_;
  Shape out_matmul_shape_;

  // Contracting dimensions of the LHS and RHS matmul shapes.
  absl::InlinedVector<int64_t, 2> lhs_matmul_contracting_dims_;
  absl::InlinedVector<int64_t, 2> rhs_matmul_contracting_dims_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_DOT_THUNK_H_
