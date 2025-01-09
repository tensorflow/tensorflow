/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/cpu/in_process_collectives.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/in_process_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu::runtime {

absl::StatusOr<std::shared_ptr<Communicator>>
InProcessCollectives::GetCommunicator(absl::Span<GlobalDeviceId const> devices,
                                      int rank) {
  absl::MutexLock lock(&mu_);

  std::shared_ptr<InProcessCommunicator::State> state = state_.lock();
  if (state == nullptr) {
    state = InProcessCommunicator::CreateState();
    state_ = state;
  }

  // We don't care about devices here: we share rendezvous state globally.
  return std::make_shared<InProcessCommunicator>(std::move(state), rank,
                                                 devices.size());
}

}  // namespace xla::cpu::runtime
