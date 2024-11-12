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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

NcclGroupThunk::NcclGroupThunk(const HloInstruction* instruction,
                               Thunk::Kind kind,
                               std::vector<std::unique_ptr<Thunk>> thunks)
    : Thunk(kind, ThunkInfo::WithProfileAnnotation(instruction)) {
  nccl_api_ = NcclApi::Default();
  for (auto& thunk : thunks) {
    thunks_.emplace_back(std::move(thunk));
  }
}
absl::Status NcclGroupThunk::Prepare(const PrepareParams& params,
                                     ResourceRequests& resource_requests) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Prepare(params, resource_requests));
  }
  return absl::OkStatus();
}
absl::Status NcclGroupThunk::Initialize(const InitializeParams& params) {
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status NcclGroupThunk::ExecuteOnStream(
    const Thunk::ExecuteParams& params) {
  TF_RETURN_IF_ERROR(nccl_api_->GroupStart());
  for (const std::unique_ptr<Thunk>& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
  }
  TF_RETURN_IF_ERROR(nccl_api_->GroupEnd());
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
