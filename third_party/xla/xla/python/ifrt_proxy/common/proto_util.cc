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
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "tsl/platform/status_to_from_proto.h"

namespace xla {
namespace ifrt {
namespace proxy {

std::unique_ptr<IfrtResponse> NewIfrtResponse(uint64_t op_id,
                                              absl::Status status) {
  auto ifrt_resp = std::make_unique<IfrtResponse>();
  auto* response_metadata = ifrt_resp->mutable_response_metadata();
  response_metadata->set_op_id(op_id);
  *response_metadata->mutable_status() = tsl::StatusToProto(status);
  return ifrt_resp;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
