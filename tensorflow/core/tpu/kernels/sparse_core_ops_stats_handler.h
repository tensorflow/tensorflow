/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_STATS_HANDLER_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_STATS_HANDLER_H_

#include <cstdint>
#include <string>

enum class StatsType {
  NUM_MINIBATCHES_PER_SC,
  MAX_IDS_PER_PARTITION,
  MAX_UNIQUE_IDS_PER_PARTITION,
  IDS_PER_PARTITION,
  UNIQUE_IDS_PER_PARTITION,
  DROPPED_ID_COUNT,
};

class SparseCoreOpsStatsHandler {
 public:
  virtual ~SparseCoreOpsStatsHandler() = default;
  virtual void Record(
      StatsType type, int64_t value, std::string device_name,
      std::string table_name) { /* Default implementation does nothing */
  }
};

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_STATS_HANDLER_H_
