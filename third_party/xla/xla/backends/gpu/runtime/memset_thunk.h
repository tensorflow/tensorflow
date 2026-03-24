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

#ifndef XLA_BACKENDS_GPU_RUNTIME_MEMSET_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_MEMSET_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

// This file contains thunks that set a buffer's elements to a particular value.
// This can be faster than emitting a kernel to set the elements.

namespace xla {
namespace gpu {

// Thunk that zeroes out a given chunk of memory.
class MemzeroThunk : public Thunk {
 public:
  explicit MemzeroThunk(ThunkInfo thunk_info, const ShapedSlice& dest)
      : Thunk(Kind::kMemzero, thunk_info), dest_(dest) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const ShapedSlice& destination() const { return dest_; }

  BufferUses buffer_uses() const override {
    return {
        BufferUse::Write(dest_.slice, dest_.shape),
    };
  }

  static absl::StatusOr<std::unique_ptr<MemzeroThunk>> FromProto(
      ThunkInfo thunk_info, const MemzeroThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const ShapedSlice dest_;
};

// Thunk that sets a given chunk of memory to a particular 32-bit value.  The
// destination chunk must have size divisible by 32 bits.
class Memset32BitValueThunk : public Thunk {
 public:
  explicit Memset32BitValueThunk(ThunkInfo thunk_info, uint32_t value,
                                 const BufferAllocation::Slice& dest)
      : Thunk(Kind::kMemset32BitValue, thunk_info),
        value_(value),
        dest_(dest) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  const BufferAllocation::Slice& destination() const { return dest_; }
  uint32_t value() const { return value_; }

  BufferUses buffer_uses() const override {
    return {
        BufferUse::Write(dest_, ShapeUtil::MakeShape(U32, {})),
    };
  }

  static absl::StatusOr<std::unique_ptr<Memset32BitValueThunk>> FromProto(
      ThunkInfo thunk_info, const Memset32BitValueThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const uint32_t value_;
  const BufferAllocation::Slice dest_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_MEMSET_THUNK_H_
