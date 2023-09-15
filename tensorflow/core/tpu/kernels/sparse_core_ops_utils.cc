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
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

std::vector<int> ConvertBinarySplitsToBucketSplits(int64 split,
                                                   int max_division_level) {
  std::vector<int> bucket_splits;
  uint32 current_index = 0;
  while (split > 0) {
    if (split % 2 == 1) {
      int split_level = absl::bit_width(current_index + 1) - 1;
      int split_offset = current_index - (1 << split_level) + 1;
      int split_size = 1 << (max_division_level - 1 - split_level);
      bucket_splits.push_back(split_size + split_offset * split_size * 2);
    }
    split >>= 1;
    current_index += 1;
  }
  absl::c_sort(bucket_splits);
  return bucket_splits;
}

int64 ConvertBucketSplitsToBinarySplits(std::vector<int> bucket_splits,
                                        int max_division_level) {
  int64 binary_splits = 0;
  for (auto& bucket_split : bucket_splits) {
    int split_level = max_division_level - 1;
    while (bucket_split > 0 && bucket_split % 2 == 0) {
      --split_level;
      bucket_split = bucket_split >> 1;
    }
    binary_splits |= (1LL << ((1 << split_level) - 1 + bucket_split / 2));
  }
  return binary_splits;
}

Status ValidateInputCombiner(const std::string& combiner) {
  if (combiner != "sum" && combiner != "mean" && combiner != "sqrtn") {
    return absl::InvalidArgumentError(
        "Invalid combiner: only \"sum\", \"mean\", and "
        "\"sqrtn\" are supported.");
  }
  return OkStatus();
}

std::function<float(float)> GetCombinerScaleContributionFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1.f; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x; };
  } else {  // combiner == "sqrtn"
    return [](float x) -> float { return x * x; };
  }
}

std::function<float(float)> GetCombinerScaleTransformFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / x; };
  } else {  // combiner == "sqrtn"
    return
        [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / std::sqrt(x); };
  }
}

ABSL_ATTRIBUTE_WEAK int GetMinibatchMaxDivisionLevel() {
  return kMinibatchMaxDivisionLevel;
}

}  // namespace tensorflow
