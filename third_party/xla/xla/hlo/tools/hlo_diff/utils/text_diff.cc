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
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
namespace xla::hlo_diff {

namespace {

// Computes LCS lengths row for s1 and s2 using O(s2.length()) space.
// If reverse == true, computes for reversed s1 and s2.
std::vector<int> GetLcsLengths(absl::string_view s1, absl::string_view s2,
                               bool reverse = false) {
  int m = s1.length();
  int n = s2.length();
  std::vector<int> prev(n + 1, 0);
  std::vector<int> curr(n + 1, 0);
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      char c1 = reverse ? s1[m - i] : s1[i - 1];
      char c2 = reverse ? s2[n - j] : s2[j - 1];
      if (c1 == c2) {
        curr[j] = prev[j - 1] + 1;
      } else {
        curr[j] = std::max(prev[j], curr[j - 1]);
      }
    }
    prev = curr;
  }
  return curr;
}

void MergeOrPushChunk(TextDiffType type, absl::string_view text,
                      std::vector<TextDiffChunk>& chunks) {
  if (text.empty()) {
    return;
  }
  if (!chunks.empty() && chunks.back().type == type) {
    chunks.back().text.append(text);
  } else {
    chunks.push_back({type, std::string(text)});
  }
}

void ComputeDiffRecursive(absl::string_view left, absl::string_view right,
                          std::vector<TextDiffChunk>& chunks) {
  int m = left.length();
  int n = right.length();

  if (m == 0) {
    MergeOrPushChunk(TextDiffType::kAdded, right, chunks);
    return;
  }
  if (n == 0) {
    MergeOrPushChunk(TextDiffType::kRemoved, left, chunks);
    return;
  }

  if (m == 1 || n == 1) {
    // Fallback to O(MN) DP for small inputs to simplify base cases.
    // With M=1 or N=1, this is efficient enough space-wise.
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
    std::vector<TextDiffChunk> reversed_chunks;
    int i = m, j = n;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && left[i - 1] == right[j - 1]) {
        if (!reversed_chunks.empty() &&
            reversed_chunks.back().type == TextDiffType::kUnchanged) {
          reversed_chunks.back().text.insert(0, 1, left[i - 1]);
        } else {
          reversed_chunks.push_back(
              {TextDiffType::kUnchanged, std::string(1, left[i - 1])});
        }
        i--;
        j--;
      } else if (i > 0 && (j == 0 || dp[i - 1][j] > dp[i][j - 1])) {
        if (!reversed_chunks.empty() &&
            reversed_chunks.back().type == TextDiffType::kRemoved) {
          reversed_chunks.back().text.insert(0, 1, left[i - 1]);
        } else {
          reversed_chunks.push_back(
              {TextDiffType::kRemoved, std::string(1, left[i - 1])});
        }
        i--;
      } else {
        if (!reversed_chunks.empty() &&
            reversed_chunks.back().type == TextDiffType::kAdded) {
          reversed_chunks.back().text.insert(0, 1, right[j - 1]);
        } else {
          reversed_chunks.push_back(
              {TextDiffType::kAdded, std::string(1, right[j - 1])});
        }
        j--;
      }
    }
    std::reverse(reversed_chunks.begin(), reversed_chunks.end());
    for (const auto& chunk : reversed_chunks) {
      MergeOrPushChunk(chunk.type, chunk.text, chunks);
    }
    return;
  }

  int mid = m / 2;
  std::vector<int> l1 = GetLcsLengths(left.substr(0, mid), right, false);
  std::vector<int> l2 = GetLcsLengths(left.substr(mid), right, true);

  int partition = 0;
  int max_lcs = -1;
  for (int j = 0; j <= n; ++j) {
    if (l1[j] + l2[n - j] > max_lcs) {
      max_lcs = l1[j] + l2[n - j];
      partition = j;
    }
  }

  ComputeDiffRecursive(left.substr(0, mid), right.substr(0, partition), chunks);
  ComputeDiffRecursive(left.substr(mid), right.substr(partition), chunks);
}

}  // namespace

std::vector<TextDiffChunk> ComputeTextDiff(absl::string_view left,
                                           absl::string_view right) {
  static absl::NoDestructor<absl::flat_hash_map<
      std::pair<std::string, std::string>, std::vector<TextDiffChunk>>>
      cache;
  std::pair<std::string, std::string> key{std::string(left),
                                          std::string(right)};
  auto it = cache->find(key);
  if (it != cache->end()) {
    return it->second;
  }
  std::vector<TextDiffChunk> diff_chunks;
  ComputeDiffRecursive(left, right, diff_chunks);
  return cache->emplace(key, diff_chunks).first->second;
}

}  // namespace xla::hlo_diff
