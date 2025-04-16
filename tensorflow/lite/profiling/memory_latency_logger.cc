/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/memory_latency_logger.h"

#include <algorithm>
#include <iomanip>
#include <ios>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {
namespace memory {

MemoryLatencyLogger::MemoryLatencyLogger() {
  mem_monitor_ =
      std::make_unique<MemoryUsageMonitor>(/*sampling_interval_ms=*/50);
}
void MemoryLatencyLogger::Start() {
  if (start_ != absl::UnixEpoch()) {
    TFLITE_LOG(INFO) << "MemoryLatencyLogger start called multiple times.";
    return;
  }
  start_ = absl::Now();
  mem_monitor_->Start();
}

void MemoryLatencyLogger::Stop(absl::string_view log_message) {
  if (start_ == absl::UnixEpoch()) {
    TFLITE_LOG(INFO)
        << "MemoryLatencyLogger hasn't started yet or has stopped!";
    return;
  }

  absl::Time stop = absl::Now();
  mem_monitor_->Stop();
  int space_count =
      35 - log_message.size();  // used for better user readability.
  std::string space(std::max(space_count, 0), '-');
  TFLITE_LOG(INFO) << log_message << " " << space << " latency: " << std::fixed
                   << std::setprecision(1)
                   << absl::ToDoubleMilliseconds(stop - start_)
                   << " ms, peak alloc: " << mem_monitor_->GetPeakMemUsageInMB()
                   << " MB, peak in-use: "
                   << mem_monitor_->GetPeakInUseMemoryInMB()
                   << " MB, current in-use: "
                   << mem_monitor_->GetCurrentInUseMemoryInMB() << " MB";
}

}  // namespace memory
}  // namespace profiling
}  // namespace tflite
