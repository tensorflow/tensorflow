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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Dot operation implemented on top of XNNPACK.
class XnnDotThunk final : public Thunk {
 public:
  ~XnnDotThunk() final;

  // Returns true if the dot operation is supported by XNNPACK. Returns an error
  // if the dot operation shape is invalid.
  static absl::StatusOr<bool> IsSupported(
      const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
      const Shape& rhs_shape, const Shape& out_shape);

  static absl::StatusOr<std::unique_ptr<XnnDotThunk>> Create(
      Info info, DotDimensionNumbers dot_dimensions,
      BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
      BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
      BufferAllocation::Slice out_buffer, Shape out_shape);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final { return DotBufferUses(dot_slices_); }

 private:
  // XNNPACK runtime instantiated for the dot operation.
  struct XnnRuntime;

  XnnDotThunk(Info info, DotDimensionNumbers dot_dimensions,
              DotSlices dot_slices, DotShape dot_shape,
              DotCanonicalDims dot_canonical_dims);

  absl::StatusOr<XnnRuntime> CreateXnnRuntime();

  DotDimensionNumbers dot_dimensions_;
  DotSlices dot_slices_;
  DotShape dot_shape_;
  DotCanonicalDims dot_canonical_dims_;

  // XLA:CPU executable can be called concurrently from multiple threads, and we
  // need to keep a pool of XNNPACK runtimes to avoid data races.
  ObjectPool<XnnRuntime> xnn_runtime_pool_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_
