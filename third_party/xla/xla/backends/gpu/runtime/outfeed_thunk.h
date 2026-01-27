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

#ifndef XLA_BACKENDS_GPU_RUNTIME_OUTFEED_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_OUTFEED_THUNK_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"

namespace xla {
namespace gpu {

// A thunk that outfeeds data. Data must be already resident on the host. This
// thunk performs a device to host copy, from the buffer allocated for the
// outfeed op, to the host location.
class OutfeedThunk : public Thunk {
 public:
  // Constructs a OutfeedThunk that copies data to the host-side
  // outfeed queue, from the buffers in the given shape tree.
  OutfeedThunk(ThunkInfo thunk_info, std::vector<ShapedSlice> source_slices);

  OutfeedThunk(const OutfeedThunk&) = delete;
  OutfeedThunk& operator=(const OutfeedThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override {
    BufferUses res;
    res.reserve(source_slices_.size());
    for (const ShapedSlice& shaped_slice : source_slices_) {
      res.push_back(BufferUse::Read(shaped_slice.slice, shaped_slice.shape));
    }
    return res;
  }

  // Deserializes an `OutfeedThunk` that will copy the data from the given
  // `source_allocations` to the host-side outfeed queue.
  // The `source_allocations` must outlive the returned `OutfeedThunk`.
  static absl::StatusOr<std::unique_ptr<OutfeedThunk>> FromProto(
      ThunkInfo thunk_info, const OutfeedThunkProto& proto,
      absl::Span<const BufferAllocation> source_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  const std::vector<ShapedSlice> source_slices_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_OUTFEED_THUNK_H_
