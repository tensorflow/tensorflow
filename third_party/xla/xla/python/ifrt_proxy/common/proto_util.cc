// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/common/proto_util.h"

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/tsl/platform/status_to_from_proto.h"

namespace xla {
namespace ifrt {
namespace proxy {

std::unique_ptr<IfrtResponse> NewIfrtResponse(
    uint64_t op_id, absl::Status status,
    absl::Span<const UserContextId> destroyed_user_contexts) {
  auto ifrt_resp = std::make_unique<IfrtResponse>();
  auto* response_metadata = ifrt_resp->mutable_response_metadata();
  response_metadata->set_op_id(op_id);
  *response_metadata->mutable_status() = tsl::StatusToProto(status);
  if (!destroyed_user_contexts.empty()) {
    response_metadata->mutable_destroyed_user_context_ids()->Reserve(
        destroyed_user_contexts.size());
    for (const auto& user_context_id : destroyed_user_contexts) {
      response_metadata->add_destroyed_user_context_ids(
          user_context_id.value());
    }
  }
  return ifrt_resp;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
