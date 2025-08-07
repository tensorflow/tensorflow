/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/tf_data_memory_logger.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_set.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/data/tfdataz_metrics.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {

namespace {
const int64_t kLogFrequencyS = 30;  // How often to log.

struct IteratorMemoryUsage {
  std::optional<std::string> dataset_name;
  int64_t memory_usage;
  std::string model_proto;
};

int64_t TotalMemoryUsage(const std::vector<IteratorMemoryUsage>& usages) {
  int64_t total_memory_usage = 0;
  for (const auto& usage : usages) {
    total_memory_usage += usage.memory_usage;
  }
  return total_memory_usage;
}

void LogDatasetMemoryUsage() {
  absl::flat_hash_set<std::shared_ptr<TfDatazMetricsCollector>>
      metric_collectors = TfDatazMetricsRegistry::GetIteratorMetricCollectors();
  std::vector<IteratorMemoryUsage> usages;
  for (const auto& metric_collector : metric_collectors) {
    int64_t total_buffered_bytes =
        metric_collector->GetModel()->output()->TotalBufferedBytes();
    model::ModelProto model_proto;
    absl::Status s = metric_collector->GetModel()->ToProto(&model_proto);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to convert model to proto: " << s;
    }
    usages.push_back(IteratorMemoryUsage{metric_collector->DatasetName(),
                                         total_buffered_bytes,
                                         model_proto.ShortDebugString()});
  }
  std::sort(usages.begin(), usages.end(), [](const auto& a, const auto& b) {
    return a.memory_usage > b.memory_usage;
  });
  VLOG(4) << "Total buffered bytes across all (" << metric_collectors.size()
          << ") tf.data iterators: "
          << strings::HumanReadableNumBytes(TotalMemoryUsage(usages));
  VLOG(4) << "Top usages: ";
  for (int i = 0; i < 5; ++i) {
    if (i >= usages.size()) {
      break;
    }
    std::string usage_string =
        strings::HumanReadableNumBytes(usages[i].memory_usage);
    if (usages[i].dataset_name.has_value()) {
      VLOG(4) << "Dataset " << usages[i].dataset_name.value() << ": "
              << usage_string;
    } else {
      VLOG(4) << "Dataset " << i << " (no name set): " << usage_string;
    }
    VLOG(5) << "Model proto: " << usages[i].model_proto;
  }
}

void MemoryLoggerThread() {
  while (true) {
    if (VLOG_IS_ON(4)) {
      LogDatasetMemoryUsage();
    }
    Env::Default()->SleepForMicroseconds(kLogFrequencyS * 1000000);
  }
}

void StartMemoryLoggerThread() {
  ThreadOptions opts;
  // This thread is never stopped.
  [[maybe_unused]] Thread* unused = Env::Default()->StartThread(
      opts, "dataset-memory-logging-thread", []() { MemoryLoggerThread(); });
}
}  // namespace

void EnsureIteratorMemoryLoggerStarted() {
  static absl::once_flag flag;
  absl::call_once(flag, StartMemoryLoggerThread);
}

}  // namespace data
}  // namespace tensorflow
