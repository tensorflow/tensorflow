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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_UTILS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_UTILS_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Pad value used for SparseCore mini batching logic.
const int32_t kXlaPadValue = std::numeric_limits<int32_t>::max();

std::vector<int> ConvertBinarySplitsToBucketSplits(int64 split,
                                                   int max_division_level);

int64 ConvertBucketSplitsToBinarySplits(std::vector<int> bucket_splits,
                                        int max_division_level);

absl::Status ValidateInputCombiner(const std::string& combiner);

std::function<float(float)> GetCombinerScaleContributionFunction(
    absl::string_view combiner);

std::function<float(float)> GetCombinerScaleTransformFunction(
    absl::string_view combiner);

// Stacks tables, so long as table have the same 'group' index. We assume that
// all tables with a given group index have the same width. Returns a list of
// list of table names, in alphabetical order.
std::vector<std::vector<std::string>> GetTableStacks(
    const std::vector<int64_t>& table_height,
    const std::vector<int64_t>& table_width,
    const std::vector<int64_t>& table_num_samples,
    const std::vector<int64_t>& table_group,
    const std::vector<std::string>& table_names, int64_t num_tpu_chips);

int GetMinibatchMaxDivisionLevel();

bool GetDisableTableStacking();

int64_t GetXlaSparseCoreStackingMemLimit();

int64_t GetXlaSparseCoreStackingTableShardLimit();

absl::Status GetMaxIdsAndUniquesExternal(const std::string& program_key,
                                         const std::string& table_name,
                                         int64_t num_samples_per_sparse_core,
                                         int64_t feature_width,
                                         int64_t* max_ids_per_partition,
                                         int64_t* max_unique_ids_per_partition);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_UTILS_H_
