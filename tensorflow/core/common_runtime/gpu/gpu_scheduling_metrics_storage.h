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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SCHEDULING_METRICS_STORAGE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SCHEDULING_METRICS_STORAGE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/tsl/framework/real_time_in_memory_metric.h"

namespace tensorflow {

// Storage class that holds all the exported in memory metrics exported by GPU
// runtime.
class GpuSchedulingMetricsStorage {
 public:
  static GpuSchedulingMetricsStorage& GetGlobalStorage();

  // Gets the metrics for estimated total GPU load.
  tsl::RealTimeInMemoryMetric<int64_t>& TotalGpuLoadNs() {
    return total_gpu_load_ns_;
  }

  const tsl::RealTimeInMemoryMetric<int64_t>& TotalGpuLoadNs() const {
    return total_gpu_load_ns_;
  }

 private:
  tsl::RealTimeInMemoryMetric<int64_t> total_gpu_load_ns_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_SCHEDULING_METRICS_STORAGE_H_
