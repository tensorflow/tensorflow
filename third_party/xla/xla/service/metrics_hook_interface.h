/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_METRICS_HOOK_INTERFACE_H_
#define XLA_SERVICE_METRICS_HOOK_INTERFACE_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/service/metrics.pb.h"

namespace xla {

// MetricsHookInterface is an abstract interface for compiler backends to record
// stages of their compilation process.

class MetricsHookInterface {
 public:
  virtual ~MetricsHookInterface() = default;
  // Used to record instance of a successful XLA compilation pass happening
  // under a stage
  virtual void RecordStagePassCount(absl::string_view stage,
                                    absl::string_view pass) const = 0;

  // Used to record instance of an error with error_status happening under an
  // XLA compilation stage and pass
  virtual void RecordStagePassError(absl::string_view stage,
                                    absl::string_view pass,
                                    absl::string_view error_status) const = 0;

  // Used to record instance of a successful XLA compilation stage that does not
  // encompass its own passes (empty pass field).
  virtual void RecordStageCount(absl::string_view stage) const = 0;

  // Used to record instance of an error with error_status happening under an
  // XLA compilation stage (empty pass field)
  virtual void RecordStageError(absl::string_view stage,
                                absl::string_view error_status) const = 0;

  // Captures metrics for a given XLA compilation stage and module_ids. The
  // `pass_metrics` can be empty if no pass specific metrics are available.
  virtual void RecordCompilationMetrics(
      CompilationLogEntry::CompilationStage stage, absl::Duration latency,
      absl::Span<const uint64_t> module_ids,
      absl::Span<const PassMetrics> pass_metrics) const = 0;
};
}  // namespace xla
#endif  // XLA_SERVICE_METRICS_HOOK_INTERFACE_H_
