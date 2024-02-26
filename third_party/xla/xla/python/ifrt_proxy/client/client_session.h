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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_SESSION_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_SESSION_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Base class that defines the interface between IFRT service protocol and the
// stream implementation that is responsible for sending requests and receiving
// responses.
//
// `ClientSession` implementation must be thread-safe.
class ClientSession {
 public:
  // `Response` represents either an `IfrtResponse` value, or an `absl::Status`
  // value corresponding to termination of the session stream. Value will never
  // be a nullptr with OK status.
  using Response = absl::StatusOr<std::shared_ptr<IfrtResponse>>;

  virtual ~ClientSession() = default;

  // Enqueues `request` to be sent via the stream; enqueued requests are sent in
  // FIFO order. The caller must ensure that `request->op_id()` is unique
  // throughout the stream's lifetime. The returned future becomes ready when a
  // response for the given op id becomes ready.
  virtual Future<Response> Enqueue(std::unique_ptr<IfrtRequest> request) = 0;

  // Terminates the `ClientSession` if it has not already been terminated.
  virtual void Finish(const absl::Status& s) {}
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_CLIENT_SESSION_H_
