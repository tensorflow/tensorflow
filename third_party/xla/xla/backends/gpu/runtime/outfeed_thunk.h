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

#include <vector>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

// A thunk that outfeeds data. Data must be already resident on the host. This
// thunk performs a device to host copy from the buffer allocated for the
// outfeed op to the host location.
class OutfeedThunk : public Thunk {
 public:
  // Constructs a OutfeedThunk that copies data to the host-side
  // outfeed queue from the buffers in the given shape tree.
  OutfeedThunk(ThunkInfo thunk_info, std::vector<ShapedSlice> source_slices);

  OutfeedThunk(const OutfeedThunk&) = delete;
  OutfeedThunk& operator=(const OutfeedThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<ShapedSlice> source_slices_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_OUTFEED_THUNK_H_
