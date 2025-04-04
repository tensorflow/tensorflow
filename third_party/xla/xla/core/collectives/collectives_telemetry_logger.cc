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

#include "xla/core/collectives/collectives_telemetry_logger.h"

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/core/collectives/collectives_telemetry_consumer_registry.h"
#include "xla/service/collective_ops_utils.h"

namespace xla {
void CollectivesTelemetryLogger::ConsumeCollectiveTelemetry(
    absl::string_view collective_name,
    std::optional<ReductionKind> reduction_kind, PrimitiveType dtype,
    size_t count, absl::Duration duration) {
  if (VLOG_IS_ON(1)) {
    auto reduction_kind_str =
        reduction_kind.has_value()
            ? ReductionKindToString(reduction_kind.value())
            : "N/A";
    VLOG(1) << "CollectivesTelemetryLogger::ConsumeCollectiveTelemetry: "
            << collective_name << " " << reduction_kind_str << " " << dtype
            << " " << count << " " << duration;
  }
}
}  // namespace xla

XLA_COLLECTIVES_TELEMETRY_CONSUMER_REGISTER(
    1, std::make_unique<xla::CollectivesTelemetryLogger>());
