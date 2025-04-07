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

#ifndef XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_CONSUMER_REGISTRY_H_
#define XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_CONSUMER_REGISTRY_H_

#include <cstdint>
#include <memory>

#include "absl/base/attributes.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/core/collectives/collectives_telemetry_consumer.h"

namespace xla {
// A registry of collective telemetry consumers registered with the current
// process.
class CollectivesTelemetryConsumerRegistry {
 public:
  static absl::Status Register(
      int32_t priority, std::unique_ptr<CollectivesTelemetryConsumer> consumer);

  static absl::StatusOr<CollectivesTelemetryConsumer*> Get();
};

}  // namespace xla

#define XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER(PRIORITY, IMPL) \
  XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER_(PRIORITY, IMPL, __COUNTER__)
#define XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER_(PRIORITY, IMPL, N) \
  XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER__(PRIORITY, IMPL, N)
#define XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER__(PRIORITY, IMPL, N)     \
  ABSL_ATTRIBUTE_UNUSED static const bool                                    \
      xla_collectives_telemetry_consumer_##N##_registered_ = [] {            \
        absl::Status status =                                                \
            ::xla::CollectivesTelemetryConsumerRegistry::Register(PRIORITY,  \
                                                                  IMPL);     \
        if (!status.ok()) {                                                  \
          LOG(ERROR) << "Failed to register telemetry consumer: " << status; \
        }                                                                    \
        return true;                                                         \
      }()

#endif  // XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_CONSUMER_REGISTRY_H_
