/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/oneccl_communicator.h"

#include <memory>

#include "oneapi/ccl.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/oneccl_errors.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/future.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<OnecclCommunicator>> OnecclCommunicator::Create(
    absl::AnyInvocable<absl::StatusOr<onecclComm_t>()> make_comm, bool is_async,
    tsl::Env& env) {
  auto f = [&make_comm]() -> absl::StatusOr<onecclComm_t> {
    ASSIGN_OR_RETURN(onecclComm_t comm, make_comm());
    // There is no need for PollUntilDone here since oneccl comm creation is
    // blocking.
    return comm;
  };
  if (!is_async) {
    ASSIGN_OR_RETURN(onecclComm_t comm, f());
    return absl::WrapUnique(new OnecclCommunicator(comm, nullptr));
  }
  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  ASSIGN_OR_RETURN(onecclComm_t comm, MakeFutureOn(*executor, f).Await());
  return absl::WrapUnique(new OnecclCommunicator(comm, std::move(executor)));
}

OnecclCommunicator::~OnecclCommunicator() {
  auto f = [this]() -> absl::Status {
    if (comm_ == nullptr) {
      VLOG(1) << "Skipping destruction; null comm_ " << *this;
      return absl::OkStatus();
    }
    VLOG(1) << "Destroy " << *this;
    return XLA_ONECCL_STATUS(onecclCommDestroy(comm_));
  };
  if (absl::Status s = Execute(f).Await(); !s.ok()) {
    LOG(ERROR) << "OnecclCommunicator::~OnecclCommunicator: " << s;
  }
}

std::string OnecclCommunicator::ToString() const {
  // comm_ should not be "touched" outside of executor_, but we are printing
  // the pointer itself and not touching the value, so this is safe.
  return absl::StrFormat("OnecclCommunicator(onecclComm_t=%p)", comm_);
}

Future<> OnecclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return executor_ ? MakeFutureOn(*executor_, std::move(f))
                   : Future<>(std::move(f)());
}
}  // namespace xla::gpu
