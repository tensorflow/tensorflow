// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_BUFFER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_BUFFER_H_

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace boosted_trees {
namespace quantiles {

// Buffering container ideally suited for scenarios where we need
// to sort and dedupe/compact fixed chunks of a stream of weighted elements.
template <typename ValueType, typename WeightType,
          typename CompareFn = std::less<ValueType>>
class WeightedQuantilesBuffer {
 public:
  struct BufferEntry {
    BufferEntry(const ValueType& v, const WeightType& w)
        : value(v), weight(w) {}
    BufferEntry() : value(0), weight(0) {}

    bool operator<(const BufferEntry& other) const {
      return kCompFn(value, other.value);
    }
    bool operator==(const BufferEntry& other) const {
      return value == other.value && weight == other.weight;
    }
    friend std::ostream& operator<<(std::ostream& strm,
                                    const BufferEntry& entry) {
      return strm << "{" << entry.value << ", " << entry.weight << "}";
    }
    ValueType value;
    WeightType weight;
  };

  explicit WeightedQuantilesBuffer(int64 block_size, int64 max_elements)
      : max_size_(std::min(block_size << 1, max_elements)) {
    QCHECK(max_size_ > 0) << "Invalid buffer specification: (" << block_size
                          << ", " << max_elements << ")";
    map_.reserve(max_size_);
  }

  // Disallow copying as it's semantically non-sensical in the Squawd algorithm
  // but enable move semantics.
  WeightedQuantilesBuffer(const WeightedQuantilesBuffer& other) = delete;
  WeightedQuantilesBuffer& operator=(const WeightedQuantilesBuffer&) = delete;
  WeightedQuantilesBuffer(WeightedQuantilesBuffer&& other) = default;
  WeightedQuantilesBuffer& operator=(WeightedQuantilesBuffer&& other) = default;

  // Push entry to buffer and maintain a compact representation within
  // pre-defined size limit.
  void PushEntry(const ValueType& value, const WeightType& weight) {
    // Callers are expected to act on a full compacted buffer after the
    // PushEntry call returns.
    QCHECK(!IsFull()) << "Buffer already full: " << max_size_;

    // Ignore zero and negative weight entries.
    if (weight <= 0) {
      return;
    }

    // Insert entry to map if not already present else
    // accumulate the new weight.
    auto result = map_.insert(BufferMapEntry(value, weight));
    if (!result.second) {
      result.first->second += weight;
    }
  }

  // Returns a sorted vector view of the base buffer. Callers should
  // minimize how often this is called, ideally only right after the buffer
  // becomes full.
  std::vector<BufferEntry> GenerateEntryList() const {
    std::vector<BufferEntry> ret;
    ret.reserve(map_.size());
    std::transform(map_.begin(), map_.end(), std::back_inserter(ret),
                   [](const BufferMapEntry& map_entry) {
                     return BufferEntry(map_entry.first, map_entry.second);
                   });
    std::sort(ret.begin(), ret.end());
    return ret;
  }

  int64 Size() const { return map_.size(); }
  bool IsFull() const { return map_.size() >= max_size_; }
  void Clear() { map_.clear(); }

 private:
  using BufferMap = typename std::unordered_map<ValueType, WeightType>;
  using BufferMapEntry = typename BufferMap::value_type;

  // Comparison function.
  static constexpr decltype(CompareFn()) kCompFn = CompareFn();

  // Base buffer.
  size_t max_size_;
  BufferMap map_;
};

template <typename ValueType, typename WeightType, typename CompareFn>
constexpr decltype(CompareFn())
    WeightedQuantilesBuffer<ValueType, WeightType, CompareFn>::kCompFn;

}  // namespace quantiles
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_BUFFER_H_
