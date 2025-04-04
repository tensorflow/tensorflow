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

#ifndef XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_LOGGER_H_
#define XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_LOGGER_H_

#include <cstddef>
#include <optional>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/core/collectives/collectives_telemetry_consumer.h"
#include "xla/service/collective_ops_utils.h"

namespace xla {

class CollectivesTelemetryLogger : public CollectivesTelemetryConsumer {
 public:
  void ConsumeCollectiveTelemetry(absl::string_view collective_name,
                                  std::optional<ReductionKind> reduction_kind,
                                  PrimitiveType dtype, size_t count,
                                  absl::Duration duration) override;
};

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_COLLECTIVES_TELEMETRY_LOGGER_H_
