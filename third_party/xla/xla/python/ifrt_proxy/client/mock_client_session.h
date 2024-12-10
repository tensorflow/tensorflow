/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_CLIENT_SESSION_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_CLIENT_SESSION_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"

namespace xla {
namespace ifrt {
namespace proxy {

class MockClientSession final : public ClientSession {
 public:
  MOCK_METHOD(Future<Response>, Enqueue, (std::unique_ptr<IfrtRequest> req),
              (override));
  MOCK_METHOD(void, Finish, (const absl::Status& s), (override));
};

MATCHER_P(IfrtRequestOfType, req_type_param, "") {
  const std::unique_ptr<IfrtRequest>& req = arg;
  const IfrtRequest::RequestCase& req_type = req_type_param;
  return req->request_case() == req_type;
}

ACTION_P(MockClientCaptureAndReturn, requests_queue_param,
         response_proto_param) {
  auto response = std::make_unique<IfrtResponse>(response_proto_param);
  TestQueue<IfrtRequest>* requests_queue = requests_queue_param;
  const std::unique_ptr<IfrtRequest>& req = arg0;
  requests_queue->Push(*req);
  response->mutable_response_metadata()->set_op_id(
      arg0->request_metadata().op_id());
  return Future<ClientSession::Response>(std::move(response));
}

ACTION_P(MockClientSessionReturnResponse, response_proto) {
  auto response = std::make_unique<IfrtResponse>(response_proto);
  response->mutable_response_metadata()->set_op_id(
      arg0->request_metadata().op_id());
  return Future<ClientSession::Response>(std::move(response));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_CLIENT_SESSION_H_
