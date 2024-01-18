/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime3/for_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

ForThunk::ForThunk(ThunkInfo thunk_info, const int64_t loop_limit,
                   std::unique_ptr<ThunkSequence> body_thunk_sequence)
    : Thunk(Kind::kFor, thunk_info),
      loop_limit_(loop_limit),
      body_thunk_sequence_(std::make_unique<SequentialThunk>(
          // Pass nullptr as the HloInstruction* to the body_thunk_sequence_
          // constructor because this SequentialThunk is logically "part of"
          // this ForThunk, and shouldn't be profiled separately from it.
          ThunkInfo(thunk_info.op), std::move(*body_thunk_sequence))) {}

absl::Status ForThunk::Initialize(const InitializeParams& params) {
  TF_RETURN_IF_ERROR(body_thunk_sequence_->Initialize(params));
  return absl::OkStatus();
}

absl::Status ForThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(2) << "Executing ForThunk with " << loop_limit_ << " iters";
  for (int64_t i = 0; i < loop_limit_; ++i) {
    VLOG(3) << "Executing iteration # " << i;
    // Invoke loop body thunk sequence.
    TF_RETURN_IF_ERROR(body_thunk_sequence_->ExecuteOnStream(params));
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
