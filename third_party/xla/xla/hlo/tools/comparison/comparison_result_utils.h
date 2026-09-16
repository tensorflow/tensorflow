/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_COMPARISON_RESULT_UTILS_H_
#define XLA_HLO_TOOLS_COMPARISON_COMPARISON_RESULT_UTILS_H_

#include <string>
#include <vector>

#include "xla/hlo/tools/comparison/comparison_result.pb.h"

namespace xla::numerics::comparison {
struct ColorThreshold {
  double value;
  std::string background_color;
};

std::vector<ColorThreshold> GetScoreThresholds();

std::string GetColorForScore(double score);

// Computes a dissimilarity score between the baseline and target tensors. The
// score is in the range of [0, 100], where 0 means the tensors are identical
// and 100 indicates maximum dissimilarity. If the comparison fails (e.g. due
// to shape mismatch, or missing data), the score will be -1.
double ComputeDiffScore(const ComparisonResultProto& result);
}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_COMPARISON_RESULT_UTILS_H_
