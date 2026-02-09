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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_

#include <vector>

#include "absl/types/span.h"

namespace tensorflow {
namespace text {

using Mask = std::vector<bool>;
template <typename T>
using Values = std::vector<T>;
template <typename T>
using ValuesSpan = absl::Span<T>;
template <typename Tsplits>
using RowSplits = std::vector<Tsplits>;
template <typename Tsplits>
using RowSplitsSpan = absl::Span<Tsplits>;

template <typename T>
class Trimmer {
  using ValuesT = Values<T>;

 public:
  // Generates masks for a single batch of values.
  virtual std::vector<Mask> GenerateMasks(
      const std::vector<ValuesT>& values) const = 0;

  // Trims a single batch of values.
  virtual void Trim(std::vector<ValuesT>* values) const = 0;

  virtual ~Trimmer() = default;
};

template <typename T, typename Tsplits>
class BatchTrimmer {
  using Values_ = Values<T>;
  using ValuesSpan_ = ValuesSpan<T>;
  using RowSplits_ = RowSplits<Tsplits>;
  using RowSplitsSpan_ = RowSplitsSpan<Tsplits>;

 public:
  // Generates masks for a batch of value row splits.
  //
  // Args:
  //   row_splits: Row splits of the values in the shape [batch, (num values)]
  //
  // Returns:
  //   The returned value is a flattened list of mask values which can be split
  //   into batches using the same input row splits.
  virtual std::vector<Mask> GenerateMasksBatch(
      const std::vector<RowSplits_>& row_splits) const = 0;
  virtual std::vector<Mask> GenerateMasksBatch(
      const std::vector<RowSplitsSpan_>& row_splits) const = 0;

  // Trims a batch of values given their flattened values and row splits.
  //
  // Args:
  //   flat_values: Flattened values in shape [batch, (num values)]
  //   row_splits: Row splits of the values in the shape [batch, (num values)]
  //
  // Returns:
  //    The returned values are the flattened trimmed values and new row splits.
  virtual std::pair<std::vector<Values_>, std::vector<RowSplits_>> TrimBatch(
      const std::vector<Values_>& flat_values,
      const std::vector<RowSplits_>& row_splits) const = 0;
  virtual std::pair<std::vector<Values_>, std::vector<RowSplits_>> TrimBatch(
      const std::vector<ValuesSpan_>& flat_values,
      const std::vector<RowSplitsSpan_>& row_splits) const = 0;

  virtual ~BatchTrimmer() = default;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_
