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

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEXT_DIFF_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEXT_DIFF_H_
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace xla::hlo_diff {

// Represents the type of a text difference chunk.
enum class TextDiffType {
  kUnchanged,
  kAdded,
  kRemoved,
};

// A single chunk of text difference.
struct TextDiffChunk {
  TextDiffType type;
  std::string text;
  // Equality operator for testing.
  bool operator==(const TextDiffChunk& other) const {
    return type == other.type && text == other.text;
  }
};

// Computes the line-based text differences between two strings.
// Returns a vector of TextDiffChunks representing the differences.
std::vector<TextDiffChunk> ComputeTextDiff(absl::string_view left,
                                           absl::string_view right);

}  // namespace xla::hlo_diff
#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_TEXT_DIFF_H_
