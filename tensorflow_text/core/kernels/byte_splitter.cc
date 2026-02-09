// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/byte_splitter.h"

#include <vector>

namespace tensorflow {
namespace text {

void ByteSplitter::Split(const absl::string_view input,
                         std::vector<unsigned char>* bytes,
                         std::vector<int32_t>* start_offsets,
                         std::vector<int32_t>* end_offsets) const {
  if (input.empty()) return;
  Split(input, bytes);
  start_offsets->push_back(0);
  for (int i = 1; i < input.size(); ++i) {
    start_offsets->push_back(i);
    end_offsets->push_back(i);
  }
  end_offsets->push_back(input.size());
}

void ByteSplitter::Split(const absl::string_view input,
                         std::vector<unsigned char>* bytes,
                         std::vector<int32_t>* offsets) const {
  if (input.empty()) return;
  Split(input, bytes);
  for (int i = 0; i <= input.size(); ++i) {
    offsets->push_back(i);
  }
}

void ByteSplitter::Split(const absl::string_view input,
                         std::vector<unsigned char>* bytes) const {
  for (const auto& c : input) {
    bytes->push_back(c);
  }
}

absl::StatusOr<std::vector<absl::string_view>> ByteSplitter::SplitByOffsets(
      absl::string_view input,
      absl::Span<const int> start_offsets,
      absl::Span<const int> end_offsets) const {
  std::vector<absl::string_view> result;
  int num = std::min(start_offsets.size(), end_offsets.size());
  for (int i = 0; i < num; ++i) {
    if (start_offsets[i] < 0 || start_offsets[i] > input.size()) {
      return absl::InvalidArgumentError("Start offsets out of range.");
    }
    if (end_offsets[i] < 0 || end_offsets[i] > input.size()) {
      return absl::InvalidArgumentError("End offsets out of range.");
    }
    if (start_offsets[i] > end_offsets[i]) {
      return absl::InvalidArgumentError("Start offset after end offset.");
    }
    result.push_back(input.substr(start_offsets[i],
                                  end_offsets[i] - start_offsets[i]));
  }
  return result;
}

}  // namespace text
}  // namespace tensorflow
