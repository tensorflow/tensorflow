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

#include "xla/service/gpu/runtime/nccl_group_thunk.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"

namespace xla {
namespace gpu {

NcclGroupThunk::NcclGroupThunk(Thunk::Kind kind, ThunkInfo thunk_info,
                               NcclApi* nccl_api,
                               const HloInstruction* instruction,
                               int64_t replica_count, int64_t partition_count)
    : NcclCollectiveThunk(kind, thunk_info, nccl_api,
                          /*is_sync=*/false),
      config_() {
  nccl_api_ = NcclApi::Default();
}

absl::Status NcclGroupThunk::RunNcclCollective(
    const ExecuteParams& params, se::Stream& stream,
    NcclCommHandleWrapper comm_wrapper) {
  return absl::UnimplementedError(
      "RunNcclCollective not implemented for NcclGroupThunk");
}

AsyncStreamKind NcclGroupThunk::GetAsyncStreamKind() const {
  return AsyncStreamKind::kCollective;
}
}  // namespace gpu
}  // namespace xla
