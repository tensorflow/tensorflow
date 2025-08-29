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

#include "xla/hlo/tools/hlo_diff/utils/text_diff.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
namespace xla::hlo_diff {

std::vector<TextDiffChunk> ComputeTextDiff(absl::string_view left,
                                           absl::string_view right) {
  int m = left.size();
  int n = right.size();

  // dp[i][j] stores the length of the LCS of left[0...i-1] and right[0...j-1]
  std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (left[i - 1] == right[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  std::vector<TextDiffChunk> diff_chunks;
  int i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && left[i - 1] == right[j - 1]) {
      // Unchanged character
      if (!diff_chunks.empty() &&
          diff_chunks.back().type == TextDiffType::kUnchanged) {
        diff_chunks.back().text.insert(0, 1, left[i - 1]);
      } else {
        diff_chunks.push_back(
            {TextDiffType::kUnchanged, std::string(1, left[i - 1])});
      }
      i--;
      j--;
    } else if (i > 0 && (j == 0 || dp[i - 1][j] > dp[i][j - 1])) {
      // Character removed from left. This path is chosen if skipping left[i-1]
      // results in a strictly longer LCS than skipping right[j-1].
      if (!diff_chunks.empty() &&
          diff_chunks.back().type == TextDiffType::kRemoved) {
        diff_chunks.back().text.insert(0, 1, left[i - 1]);
      } else {
        diff_chunks.push_back(
            {TextDiffType::kRemoved, std::string(1, left[i - 1])});
      }
      i--;
    } else {  // j > 0, implies dp[i][j - 1] >= dp[i - 1][j] or i == 0.
      // Character added to right. This includes the tie-breaking case
      // dp[i][j - 1] == dp[i - 1][j], ensuring Added comes after Removed
      // in the final reversed list.
      if (!diff_chunks.empty() &&
          diff_chunks.back().type == TextDiffType::kAdded) {
        diff_chunks.back().text.insert(0, 1, right[j - 1]);
      } else {
        diff_chunks.push_back(
            {TextDiffType::kAdded, std::string(1, right[j - 1])});
      }
      j--;
    }
  }
  std::reverse(diff_chunks.begin(), diff_chunks.end());
  return diff_chunks;
}

}  // namespace xla::hlo_diff
