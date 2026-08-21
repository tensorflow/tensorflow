/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_COST_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_COST_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/cost_constants.h"
#include "tensorflow/core/common_runtime/cost_measurement.h"
#include "tensorflow/core/common_runtime/request_cost.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_stats.h"

namespace tensorflow {
namespace serving {

// Records batch cost to the /tensorflow/serving/batching/costs Sampler metric.
void RecordBatchCosts(const std::string& model_name, int64_t processed_size,
                      absl::string_view cost_type, absl::Duration total_cost);

// Splits the batch costs to each task.
//
// Inputs:
// 1) batch_cost_measurements, which provides the total cost of each type;
// 2) processed_size, it's the batch size plus the padding amount;
// 3) batch, provides the batch size and input sizes.
//
// Outputs:
// The request_cost in each batch task will be updated.
// - This function will use two approaches to split the batch cost (if it's
//   non-zero), thus two costs will be output.
//   1) smeared cost: batch cost is split proportionally to each task's size,
//      and paddings do not share any cost;
//   2) non-smeared cost: batch cost is split proportionally to each task or
//      padding's size. Here padding's cost is not assigned to any tasks.
// - This function will also record the metrics of this batch in each task,
//   including:
//   1) the batch size;
//   2) the input size from this task;
//   3) the padding amount.
template <typename TaskType>
void SplitBatchCostsAndRecordMetrics(
    const std::string& model_name, const std::string& op_name,
    const std::vector<std::unique_ptr<CostMeasurement>>&
        batch_cost_measurements,
    const int64_t processed_size, Batch<TaskType>& batch) {
  absl::flat_hash_map<std::string, absl::Duration> batch_costs;
  // 1. Split the batch costs to each task.
  for (const auto& batch_cost_measurement : batch_cost_measurements) {
    if (batch_cost_measurement->GetTotalCost() <= absl::ZeroDuration()) {
      continue;
    }
    if (batch.size() == 0) {  // NOLINT: empty() checks the batch contains 0
                              // tasks. size() gets the sum of task sizes.
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the batch size is 0.";
      return;
    }
    if (processed_size == 0) {
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the processed size is 0.";
      return;
    }
    const absl::string_view cost_type = batch_cost_measurement->GetCostType();
    const absl::Duration total_cost = batch_cost_measurement->GetTotalCost();
    batch_costs[cost_type] = total_cost;

    // Smeared batch cost: cost for processing this batch.
    RecordBatchCosts(model_name, processed_size,
                     absl::StrCat(cost_type, kWithSmearSuffix), total_cost);
    // Non-smeared batch cost: cost for processing inputs in this batch, i.e.
    // cost for processing paddings is excluded.
    RecordBatchCosts(model_name, processed_size,
                     absl::StrCat(cost_type, kNoSmearSuffix),
                     total_cost / processed_size * batch.size());

    // Register batch stats for in-process use.
    if (cost_type == kTpuCostName) {
      ModelBatchStats& model_stats = GlobalBatchStatsRegistry().model(
          /* model_name= */ model_name, /* op_name= */ op_name);
      model_stats.batch_size(processed_size).tpu_cost().Register(total_cost);
      // batch.size() is the size of the original batch before padding.
      model_stats.RegisterProcessedSize(batch.size());
    }

    for (int i = 0; i < batch.num_tasks(); i++) {
      RequestCost* request_cost = batch.task(i).request_cost;
      // Skip recording the cost if the request_cost is null.
      if (!request_cost) continue;

      // Smeared cost: cost of paddings are assigned to each task.
      const auto cost_with_smear =
          total_cost / batch.size() * batch.task(i).size();

      // Non-smeared cost: cost of paddings are not assigned to any tasks.
      const auto cost_no_smear =
          total_cost / processed_size * batch.task(i).size();

      request_cost->RecordCost(
          {{absl::StrCat(cost_type, kWithSmearSuffix), cost_with_smear},
           {absl::StrCat(cost_type, kNoSmearSuffix), cost_no_smear}});
    }
  }

  // 2. Records the batch metrics in each task.
  const int64_t padding_size = processed_size - batch.size();
  for (int i = 0; i < batch.num_tasks(); i++) {
    RequestCost* request_cost = batch.task(i).request_cost;
    // Skip recording the metrics if the request_cost is null.
    if (!request_cost) continue;

    request_cost->RecordBatchMetrics(RequestCost::BatchMetrics{
        processed_size, static_cast<int64_t>(batch.task(i).size()),
        padding_size, batch_costs});
  }
}

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCHING_UTIL_BATCH_COST_UTILS_H_
