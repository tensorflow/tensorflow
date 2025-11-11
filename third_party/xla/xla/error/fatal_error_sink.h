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

#ifndef XLA_ERROR_FATAL_ERROR_SINK_H_
#define XLA_ERROR_FATAL_ERROR_SINK_H_

#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"

namespace xla::error {

// An absl::LogSink that selectively handles FATAL errors to add additional
// XLA-specific debugging context.
class FatalErrorSink : public absl::LogSink {
 public:
  ~FatalErrorSink() override;

  void Send(const absl::LogEntry& entry) override;
};

// Adds the FatalErrorSink as an Absl::LogSink.
// This is thread-safe and idempotent - can be called multiple times.
void AddFatalErrorSink();

}  // namespace xla::error

#endif  // XLA_ERROR_FATAL_ERROR_SINK_H_
