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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_SESSION_HANDLER_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_SESSION_HANDLER_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"

namespace xla {
namespace ifrt {
namespace proxy {

// IfrtSessionHandler glues an incoming stream to a stack of backend runtimes
// abstracted out by a `BackendInterface`. It utilizes the provided `Backend` to
// process the incoming client requests after ensuring that dependencies as
// specified by the client are honored and the chunked requests are fully
// re-assembled.
class IfrtSessionHandler {
 public:
  using BackendFactory =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<BackendInterface>>(
          uint64_t session_id)>;

  using Response = BackendInterface::Response;

  // Makes a new IfrtSessionHandler with the given Session ID that uniquely
  // identifies this session. The backend_factory cannot be a nullptr.
  static absl::StatusOr<std::unique_ptr<IfrtSessionHandler>> Create(
      uint64_t id, BackendFactory backend_factory);

  uint64_t session_id() const { return session_id_; }

  // Top-level handler the transport implementation calls to hand off a new
  // incoming request. `on_done` is called asynchronously to return responses.
  void NewIncomingRequest(
      std::unique_ptr<IfrtRequest> request,
      std::function<void(std::shared_ptr<IfrtResponse>)> on_done);

 private:
  IfrtSessionHandler(uint64_t id, BackendFactory backend_factory);

  // InitRequest is treated somewhat differently than the rest since it triggers
  // the creation of the backend_
  Future<Response> ProcessInitRequest(std::unique_ptr<IfrtRequest> request)
      ABSL_LOCKS_EXCLUDED(backend_mu_);

  // Sets up the backaned_ only if needed - i.e., only if it is a nullptr.
  absl::Status SetupBackendIfNeeded() ABSL_LOCKS_EXCLUDED(backend_mu_);

  const uint64_t session_id_;  // Unique ID of this Session.

  // The backend_ runtime(s) this session relies on for processing the incoming
  // requests. It is instantiated at the start of a new Bidi stream, and
  // currently does not change for the life of this object.
  BackendFactory backend_factory_;
  absl::Mutex backend_mu_;
  std::unique_ptr<BackendInterface> backend_ ABSL_GUARDED_BY(backend_mu_);
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_SESSION_HANDLER_H_
