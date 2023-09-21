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
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

constexpr int kMinibatchMaxDivisionLevel = 6;

// Pad value used for SparseCore mini batching logic.
const int32_t kXlaPadValue = std::numeric_limits<int32_t>::max();

std::vector<int> ConvertBinarySplitsToBucketSplits(int64 split,
                                                   int max_division_level);

int64 ConvertBucketSplitsToBinarySplits(std::vector<int> bucket_splits,
                                        int max_division_level);

Status ValidateInputCombiner(const std::string& combiner);

std::function<float(float)> GetCombinerScaleContributionFunction(
    absl::string_view combiner);

std::function<float(float)> GetCombinerScaleTransformFunction(
    absl::string_view combiner);

int GetMinibatchMaxDivisionLevel();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_SPARSE_CORE_OPS_UTILS_H_
