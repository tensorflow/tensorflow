/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/core/collectives/collectives_telemetry_consumer_registry.h"

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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/core/collectives/collectives_telemetry_consumer.h"
#include "xla/util.h"

namespace xla {
namespace {

struct Registration {
  int32_t priority;
  std::unique_ptr<CollectivesTelemetryConsumer> consumer;
};

struct Registry {
  absl::Mutex mu;

  // Container for registered collectives implementations.
  std::optional<Registration> consumer ABSL_GUARDED_BY(mu);
};

}  // namespace

static Registry& GetConsumersRegistry() {
  static auto* const registry = new Registry();
  return *registry;
}

absl::Status CollectivesTelemetryConsumerRegistry::Register(
    int32_t priority, std::unique_ptr<CollectivesTelemetryConsumer> consumer) {
  auto& registry = GetConsumersRegistry();
  absl::MutexLock lock(&registry.mu);

  if (!registry.consumer.has_value()) {
    registry.consumer = Registration{priority, std::move(consumer)};
    return absl::OkStatus();
  }

  if (priority > registry.consumer->priority) {
    registry.consumer->priority = priority;
    registry.consumer->consumer = std::move(consumer);
  }

  return absl::OkStatus();
}

absl::StatusOr<CollectivesTelemetryConsumer*>
CollectivesTelemetryConsumerRegistry::Get() {
  auto& registry = GetConsumersRegistry();
  absl::MutexLock lock(&registry.mu);

  if (!registry.consumer.has_value()) {
    return Internal("No telemetry consumers registered");
  }

  return registry.consumer->consumer.get();
}

}  // namespace xla
