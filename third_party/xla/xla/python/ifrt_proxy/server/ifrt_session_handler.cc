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

#include "xla/python/ifrt_proxy/server/ifrt_session_handler.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"

// The tsl include below is needed only for the Status macros such as
// ASSIGN_OR_RETURN, since the OSS absl package does not have the counterparts
// yet.
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

absl::StatusOr<std::unique_ptr<IfrtSessionHandler>> IfrtSessionHandler::Create(
    uint64_t id, BackendFactory backend_factory) {
  if (backend_factory == nullptr) {
    return absl::InvalidArgumentError("BackendFactory cannot be nullptr.");
  }
  return absl::WrapUnique(
      new IfrtSessionHandler(id, std::move(backend_factory)));
}

IfrtSessionHandler::IfrtSessionHandler(uint64_t id,
                                       BackendFactory backend_factory)
    : session_id_(id), backend_factory_(std::move(backend_factory)) {}

void IfrtSessionHandler::NewIncomingRequest(
    std::unique_ptr<IfrtRequest> request,
    std::function<void(std::shared_ptr<IfrtResponse>)> on_done) {
  VLOG(2) << "NewIncomingRequest: " << request->DebugString();

  const uint64_t op_id = request->request_metadata().op_id();

  // The current implementation exploits the async nature of the backend_ IFRT
  // client to minimize the amount of work we do per request. However, using a
  // threadpool here might make sense as a performance optimization.

  auto result = [&]() -> Future<Response> {
    if (request->has_init_request()) {
      return ProcessInitRequest(std::move(request));
    }
    if (auto status = SetupBackendIfNeeded(); !status.ok()) {
      return Future<Response>(status);
    }
    absl::ReaderMutexLock read_lock(&backend_mu_);
    return backend_->Process(std::move(request));
  }();

  // Consider maintaining a count of in-flight requests (that won't complete
  // until the following OnReady callback happens) so we can safely deleting the
  // reactor_.
  result.OnReady([op_id, on_done = std::move(on_done)](
                     absl::StatusOr<std::shared_ptr<IfrtResponse>> result) {
    if (result.ok()) {
      on_done(*std::move(result));
    } else {
      on_done(NewIfrtResponse(op_id, result.status()));
    }
  });
}

Future<IfrtSessionHandler::Response> IfrtSessionHandler::ProcessInitRequest(
    std::unique_ptr<IfrtRequest> request) {
  absl::MutexLock lock(&backend_mu_);
  if (backend_ != nullptr) {
    // Currently backends cannot be reinitialized.
    return Future<Response>(absl::FailedPreconditionError(
        "This session has already been initialized."));
  }

  auto backend = backend_factory_(session_id_);
  if (!backend.ok()) {
    return Future<Response>(backend.status());
  }
  backend_ = *std::move(backend);

  return backend_->Process(std::move(request));
}

absl::Status IfrtSessionHandler::SetupBackendIfNeeded() {
  absl::MutexLock lock(&backend_mu_);
  if (backend_ != nullptr) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(backend_, backend_factory_(session_id_));
  return absl::OkStatus();
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
