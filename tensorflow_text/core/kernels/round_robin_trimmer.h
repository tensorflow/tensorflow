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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "tensorflow_text/core/kernels/trimmer.h"


namespace tensorflow {
namespace text {

template <typename T, typename Tsplits = int32_t>
class RoundRobinTrimmer : Trimmer<T>, BatchTrimmer<T, Tsplits> {
  using Values_ = Values<T>;
  using ValuesSpan_ = ValuesSpan<T>;
  using RowSplits_ = RowSplits<Tsplits>;
  using RowSplitsSpan_ = RowSplitsSpan<Tsplits>;

 public:
  RoundRobinTrimmer(int max_sequence_length)
      : max_sequence_length_(std::max(max_sequence_length, 0)) {}
  virtual ~RoundRobinTrimmer() = default;

  // Generates masks for a single batch of values.
  std::vector<Mask> GenerateMasks(
      const std::vector<Values_>& values) const;

  // Generates masks for a batch of values row splits.
  //
  // Args:
  //   row_splits: Row splits of the values in the shape [batch, (num values)]
  //
  // Returns:
  //   The returned value is a flattened list of mask values which can be split
  //   into batches using the same input row splits.
  std::vector<Mask> GenerateMasksBatch(
      const std::vector<RowSplits_>& row_splits) const;
  std::vector<Mask> GenerateMasksBatch(
      const std::vector<RowSplitsSpan_>& row_splits) const;

  // Trims a single batch of values.
  void Trim(std::vector<Values_>* values) const;

  // Trims a batch of values given their flattened values and row splits.
  //
  // Args:
  //   flat_values: Flattened values in shape [batch, (num values)]
  //   row_splits: Row splits of the values in the shape [batch, (num values)]
  //
  // Returns:
  //    The returned values are the flattened trimmed values and new row splits.
  std::pair<std::vector<Values_>, std::vector<RowSplits_>> TrimBatch(
      const std::vector<Values_>& flat_values,
      const std::vector<RowSplits_>& row_splits) const;
  std::pair<std::vector<Values_>, std::vector<RowSplits_>> TrimBatch(
      const std::vector<ValuesSpan_>& flat_values,
      const std::vector<RowSplitsSpan_>& row_splits) const;

 protected:
  // Used for holding data about value sizes and how much of it is used.
  struct Row {
    Row() : idx(0), size(0), used(0) {}
    Row(int idx, int size, int used) : idx(idx), size(size), used(used) {}
    int idx;       // Index into the list of values
    Tsplits size;  // Size of the row values
    int used;      // How much of the values is used
  };

  // Internal execution to share code for Span & Vector row_splits.
  template <typename Iterator>
  std::vector<Mask> GenerateMasksInternal(Iterator begin, Iterator end) const;

  // Internal execution to share code for Span & Vector row_splits.
  template <typename ValuesIterator, typename RowSplitsIterator>
  std::pair<std::vector<Values_>, std::vector<RowSplits_>> TrimInternal(
    ValuesIterator flat_values_begin,
    ValuesIterator flat_values_end,
    RowSplitsIterator row_splits_begin,
    RowSplitsIterator row_splits_end) const;

  // Main process of the timmer. Process row splits a batch at a time. Once each
  // it is known how much each row in a batch is used, the callback is called
  // with the row information.
  // Algorithm to fill values:
  // 1. Fill values that will max starting from smallest to largest.
  // 2. Partially fill the rest up the same amount up to the sequence length.
  // 3. Add the remainder to the available rows in order.
  template <typename Iterator>
  void ProcessBatch(Iterator values_begin, Iterator values_end,
      std::function<void(std::vector<Row>*)> callback) const;
  void ProcessBatch(std::vector<Row>* value_row_sizes,
      std::function<void(std::vector<Row>*)> callback) const;

  template <typename Iterator>
  void ProcessSplitsByBatch(Iterator begin, Iterator end,
      std::function<void(std::vector<Row>*)> callback) const;

  const int max_sequence_length_;
};

/******************************* Implementation *******************************/

template <typename T, typename Tsplits>
std::vector<Mask> RoundRobinTrimmer<T, Tsplits>::GenerateMasks(
    const std::vector<Values_>& values) const {
  std::vector<Mask> masks(values.size());
  ProcessBatch(values.begin(), values.end(),
               [&masks](std::vector<Row>* value_row_sizes) {
    for (int i = 0; i < masks.size(); ++i) {
      Mask& mask = masks[i];
      const Row& values_row = (*value_row_sizes)[i];
      mask.reserve(values_row.size);
      mask.insert(mask.end(), values_row.used, true);
      mask.insert(mask.end(), values_row.size - values_row.used, false);
    }
  });
  return masks;
}

template <typename T, typename Tsplits>
std::vector<Mask> RoundRobinTrimmer<T, Tsplits>::GenerateMasksBatch(
    const std::vector<RowSplits_>& row_splits) const {
  return GenerateMasksInternal(row_splits.begin(), row_splits.end());
}

template <typename T, typename Tsplits>
std::vector<Mask> RoundRobinTrimmer<T, Tsplits>::GenerateMasksBatch(
    const std::vector<RowSplitsSpan_>& row_splits) const {
  return GenerateMasksInternal(row_splits.begin(), row_splits.end());
}

template <typename T, typename Tsplits>
template <typename Iterator>
std::vector<Mask> RoundRobinTrimmer<T, Tsplits>::GenerateMasksInternal(
    const Iterator begin, const Iterator end) const {
  // First reserve necessary space for the masks
  std::vector<Mask> masks(end - begin);
  auto m = masks.begin();
  for (auto it = begin; it != end; ++it, ++m) {
    m->reserve(it->back());
  }
  // Process all batches, updating the masks a batch at a time.
  ProcessSplitsByBatch(begin, end, [&masks](std::vector<Row>* rows) {
    for (int s = 0; s < masks.size(); ++s) {
      const Row& row = (*rows)[s];
      masks[s].reserve(row.size);
      masks[s].insert(masks[s].end(), row.used, true);
      masks[s].insert(masks[s].end(), row.size - row.used, false);
    }
  });
  return masks;
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::Trim(std::vector<Values_>* values) const {
  ProcessBatch(values->begin(), values->end(),
               [values] (std::vector<Row>* value_row_sizes) {
    for (int s = 0; s < values->size(); ++s) {
      (*values)[s].resize((*value_row_sizes)[s].used);
    }
  });
}

template <typename T, typename Tsplits>
std::pair<std::vector<Values<T>>, std::vector<RowSplits<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::TrimBatch(
    const std::vector<Values_>& flat_values,
    const std::vector<RowSplits_>& row_splits) const {
  return TrimInternal(
      flat_values.begin(), flat_values.end(),
      row_splits.begin(), row_splits.end());
}

template <typename T, typename Tsplits>
std::pair<std::vector<Values<T>>, std::vector<RowSplits<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::TrimBatch(
    const std::vector<ValuesSpan_>& flat_values,
    const std::vector<RowSplitsSpan_>& row_splits) const {
  return TrimInternal(
      flat_values.begin(), flat_values.end(),
      row_splits.begin(), row_splits.end());
}

template <typename T, typename Tsplits>
template <typename ValuesIterator, typename RowSplitsIterator>
std::pair<std::vector<Values<T>>, std::vector<RowSplits<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::TrimInternal(
    ValuesIterator flat_values_begin,
    ValuesIterator flat_values_end,
    RowSplitsIterator splits_begin,
    RowSplitsIterator splits_end) const {
  std::pair<std::vector<Values_>, std::vector<RowSplits_>> trimmed(
      {std::vector<Values_>(flat_values_end - flat_values_begin),
       std::vector<RowSplits_>(splits_end - splits_begin)});
  // All row splits start at index 0
  for (int i = 0; i < trimmed.second.size(); ++i) {
    trimmed.second[i].push_back({0});
  }
  ProcessSplitsByBatch(splits_begin, splits_end,
      [&trimmed, flat_values_begin, splits_begin](std::vector<Row>* values_row)
  {
    auto values_it = flat_values_begin;
    auto splits_it = splits_begin;
    for (int s = 0; s < values_row->size(); ++s, ++values_it, ++splits_it) {
      Values_* vals = &trimmed.first[s];
      RowSplits_* splits = &trimmed.second[s];
      auto start = values_it->begin() + (*splits_it)[splits->size()-1];
      vals->insert(vals->end(), start, start + (*values_row)[s].used);
      splits->insert(splits->end(), splits->back() + (*values_row)[s].used);
    }
  });
  return trimmed;
}

template <typename T, typename Tsplits>
template <typename Iterator>
void RoundRobinTrimmer<T, Tsplits>::ProcessBatch(
    Iterator values_begin, Iterator values_end,
    std::function<void(std::vector<Row>*)> callback) const {
  int num_values = values_end - values_begin;
  // Get size of each segment
  std::vector<Row> value_row_sizes(num_values);
  int i = 0;
  for (auto it = values_begin; it != values_end; ++it, ++i) {
    value_row_sizes[i].idx = i;
    value_row_sizes[i].size = it->size();
  }
  // Process the values
  ProcessBatch(&value_row_sizes, callback);
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::ProcessBatch(
    std::vector<Row>* value_row_sizes,
    std::function<void(std::vector<Row>*)> callback) const {
  int num_values = value_row_sizes->size();
  int sequence_left = max_sequence_length_;

  // Fill all values to the max (smallest first to largest) that we can
  // without crossing the max_sequence_length
  std::sort(value_row_sizes->begin(), value_row_sizes->end(),
            [] (Row a, Row b) { return a.size < b.size; });
  int filled_value_rows = 0;
  for (int i = 0; i < num_values; ++i) {
    // Break if we will not be able to fill up the smallest unfilled value row
    if ((*value_row_sizes)[i].size * (num_values - filled_value_rows)
        > sequence_left) {
      break;
    }
    (*value_row_sizes)[i].used = (*value_row_sizes)[i].size;
    sequence_left -= (*value_row_sizes)[i].used;
    ++filled_value_rows;
  }

  // Fill the remaining value rows evenly
  if (filled_value_rows < num_values) {
    int count = sequence_left / (num_values - filled_value_rows);
    for (int i = filled_value_rows; i < num_values; ++i) {
      (*value_row_sizes)[i].used = count;
      sequence_left -= count;
    }
  }

  // Finally add the remainder - index order
  std::sort(value_row_sizes->begin(), value_row_sizes->end(),
            [] (Row a, Row b) { return a.idx < b.idx; });
  for (int i = 0; i < num_values && sequence_left > 0; ++i) {
    if ((*value_row_sizes)[i].used < (*value_row_sizes)[i].size) {
      ++((*value_row_sizes)[i].used);
      --sequence_left;
    }
  }

  // Usage of rows computed. Execute callback to process.
  callback(value_row_sizes);
}

template <typename T, typename Tsplits>
template <typename Iterator>
void RoundRobinTrimmer<T, Tsplits>::ProcessSplitsByBatch(
    Iterator begin, Iterator end,
    std::function<void(std::vector<Row>*)> callback) const {
  int num_in_batch = begin->size() - 1;
  int num_values = end - begin;
  // Process one batch at a time.
  std::vector<Row> value_row_sizes(num_values);
  for (int batch_idx = 0; batch_idx < num_in_batch; ++batch_idx) {
    // First, get size of each row.
    int idx = 0;
    for (auto i = begin; i < end; ++i, ++idx) {
      value_row_sizes[idx].idx = idx;
      value_row_sizes[idx].size = (*i)[batch_idx + 1] - (*i)[batch_idx];
    }
    // Perform the main processing of the batch
    ProcessBatch(&value_row_sizes, callback);
  }
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_
