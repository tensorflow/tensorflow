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

#ifndef XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_
#define XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/in_process_communicator.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu::runtime {

class InProcessCollectives : public CollectivesInterface {
 public:
  // Thread-safe.
  absl::StatusOr<std::shared_ptr<Communicator>> GetCommunicator(
      absl::Span<GlobalDeviceId const> devices, int rank) override;

 private:
  absl::Mutex mu_;

  // State shared by all constructed communicators.
  std::weak_ptr<InProcessCommunicator::State> state_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::cpu::runtime

#endif  // XLA_SERVICE_CPU_IN_PROCESS_COLLECTIVES_H_
