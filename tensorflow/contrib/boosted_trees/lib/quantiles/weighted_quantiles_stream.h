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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_STREAM_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_STREAM_H_

#include <memory>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_buffer.h"
#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_summary.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace boosted_trees {
namespace quantiles {

// Class to compute approximate quantiles with error bound guarantees for
// weighted data sets.
// This implementation is an adaptation of techniques from the following papers:
// * (2001) Space-efficient online computation of quantile summaries.
// * (2004) Power-conserving computation of order-statistics over
//          sensor networks.
// * (2007) A fast algorithm for approximate quantiles in high speed
//          data streams.
// * (2016) XGBoost: A Scalable Tree Boosting System.
//
// The key ideas at play are the following:
// - Maintain an in-memory multi-level quantile summary in a way to guarantee
//   a maximum approximation error of eps * W per bucket where W is the total
//   weight across all points in the input dataset.
// - Two base operations are defined: MERGE and COMPRESS. MERGE combines two
//   summaries guaranteeing a epsNew = max(eps1, eps2). COMPRESS compresses
//   a summary to b + 1 elements guaranteeing epsNew = epsOld + 1/b.
// - b * sizeof(summary entry) must ideally be small enough to fit in an
//   average CPU L2 cache.
// - To distribute this algorithm with maintaining error bounds, we need
//   the worker-computed summaries to have no more than eps / h error
//   where h is the height of the distributed computation graph which
//   is 2 for an MR with no combiner.
//
// We mainly want to max out IO bw by ensuring we're not compute-bound and
// using a reasonable amount of RAM.
//
// Complexity:
// Compute: O(n * log(1/eps * log(eps * n))).
// Memory: O(1/eps * log^2(eps * n)) <- for one worker streaming through the
//                                      entire dataset.
template <typename ValueType, typename WeightType,
          typename CompareFn = std::less<ValueType>>
class WeightedQuantilesStream {
 public:
  using Buffer = WeightedQuantilesBuffer<ValueType, WeightType, CompareFn>;
  using BufferEntry = typename Buffer::BufferEntry;
  using Summary = WeightedQuantilesSummary<ValueType, WeightType, CompareFn>;
  using SummaryEntry = typename Summary::SummaryEntry;

  explicit WeightedQuantilesStream(double eps, int64 max_elements)
      : eps_(eps), buffer_(1LL, 2LL), finalized_(false) {
    std::tie(max_levels_, block_size_) = GetQuantileSpecs(eps, max_elements);
    buffer_ = Buffer(block_size_, max_elements);
    summary_levels_.reserve(max_levels_);
  }

  // Disallow copy and assign but enable move semantics for the stream.
  WeightedQuantilesStream(const WeightedQuantilesStream& other) = delete;
  WeightedQuantilesStream& operator=(const WeightedQuantilesStream&) = delete;
  WeightedQuantilesStream(WeightedQuantilesStream&& other) = default;
  WeightedQuantilesStream& operator=(WeightedQuantilesStream&& other) = default;

  // Pushes one entry while maintaining approximation error invariants.
  void PushEntry(const ValueType& value, const WeightType& weight) {
    // Validate state.
    QCHECK(!finalized_) << "Finalize() already called.";

    // Push element to base buffer.
    buffer_.PushEntry(value, weight);

    // When compacted buffer is full we need to compress
    // and push weighted quantile summary up the level chain.
    if (buffer_.IsFull()) {
      PushBuffer(buffer_);
    }
  }

  // Pushes full buffer while maintaining approximation error invariants.
  void PushBuffer(Buffer& buffer) {
    // Validate state.
    QCHECK(!finalized_) << "Finalize() already called.";

    // Create local compressed summary and propagate.
    local_summary_.BuildFromBufferEntries(buffer.GenerateEntryList());
    local_summary_.Compress(block_size_, eps_);
    PropagateLocalSummary();
  }

  // Pushes full summary while maintaining approximation error invariants.
  void PushSummary(const std::vector<SummaryEntry>& summary) {
    // Validate state.
    QCHECK(!finalized_) << "Finalize() already called.";

    // Create local compressed summary and propagate.
    local_summary_.BuildFromSummaryEntries(summary);
    local_summary_.Compress(block_size_, eps_);
    PropagateLocalSummary();
  }

  // Flushes approximator and finalizes state.
  void Finalize() {
    // Validate state.
    QCHECK(!finalized_) << "Finalize() may only be called once.";

    // Flush any remaining buffer elements.
    PushBuffer(buffer_);

    // Create final merged summary.
    local_summary_.Clear();
    for (auto& summary : summary_levels_) {
      local_summary_.Merge(summary);
      summary.Clear();
    }
    summary_levels_.clear();
    summary_levels_.shrink_to_fit();
    finalized_ = true;
  }

  // Generates requested number of quantiles after finalizing stream.
  // The returned quantiles can be queried using std::lower_bound to get
  // the bucket for a given value.
  std::vector<ValueType> GenerateQuantiles(int64 num_quantiles) const {
    // Validate state.
    QCHECK(finalized_)
        << "Finalize() must be called before generating quantiles.";
    return local_summary_.GenerateQuantiles(num_quantiles);
  }

  // Generates requested number of boundaries after finalizing stream.
  // The returned boundaries can be queried using std::lower_bound to get
  // the bucket for a given value.
  // The boundaries, while still guaranteeing approximation bounds, don't
  // necessarily represent the actual quantiles of the distribution.
  // Boundaries are preferable over quantiles when the caller is less
  // interested in the actual quantiles distribution and more interested in
  // getting a representative sample of boundary values.
  std::vector<ValueType> GenerateBoundaries(int64 num_boundaries) const {
    // Validate state.
    QCHECK(finalized_)
        << "Finalize() must be called before generating boundaries.";
    return local_summary_.GenerateBoundaries(num_boundaries);
  }

  // Calculates approximation error for the specified level.
  // If the passed level is negative, the approximation error for the entire
  // summary is returned. Note that after Finalize is called, only the overall
  // error is available.
  WeightType ApproximationError(int64 level = -1) const {
    if (finalized_) {
      QCHECK(level <= 0) << "Only overall error is available after Finalize()";
      return local_summary_.ApproximationError();
    }

    if (summary_levels_.empty()) {
      // No error even if base buffer isn't empty.
      return 0;
    }

    // If level is negative, we get the approximation error
    // for the top-most level which is the max approximation error
    // in all summaries by construction.
    if (level < 0) {
      level = summary_levels_.size() - 1;
    }
    QCHECK(level < summary_levels_.size()) << "Invalid level.";
    return summary_levels_[level].ApproximationError();
  }

  size_t MaxDepth() const { return summary_levels_.size(); }

  // Generates requested number of quantiles after finalizing stream.
  const Summary& GetFinalSummary() const {
    // Validate state.
    QCHECK(finalized_)
        << "Finalize() must be called before requesting final summary.";
    return local_summary_;
  }

  // Helper method which, given the desired approximation error
  // and an upper bound on the number of elements, computes the optimal
  // number of levels and block size and returns them in the tuple.
  static std::tuple<int64, int64> GetQuantileSpecs(double eps,
                                                   int64 max_elements);

  // Serializes the internal state of the stream.
  std::vector<Summary> SerializeInternalSummaries() const {
    // The buffer should be empty for serialize to work.
    QCHECK_EQ(buffer_.Size(), 0);
    std::vector<Summary> result;
    result.reserve(summary_levels_.size() + 1);
    for (const Summary& summary : summary_levels_) {
      result.push_back(summary);
    }
    result.push_back(local_summary_);
    return result;
  }

  // Resets the state of the stream with a serialized state.
  void DeserializeInternalSummaries(const std::vector<Summary>& summaries) {
    // Clear the state before deserializing.
    buffer_.Clear();
    summary_levels_.clear();
    local_summary_.Clear();
    QCHECK_GT(max_levels_, summaries.size() - 1);
    for (int i = 0; i < summaries.size() - 1; ++i) {
      summary_levels_.push_back(summaries[i]);
    }
    local_summary_ = summaries[summaries.size() - 1];
  }

 private:
  // Propagates local summary through summary levels while maintaining
  // approximation error invariants.
  void PropagateLocalSummary() {
    // Validate state.
    QCHECK(!finalized_) << "Finalize() already called.";

    // No-op if there's nothing to add.
    if (local_summary_.Size() <= 0) {
      return;
    }

    // Propagate summary through levels.
    size_t level = 0;
    for (bool settled = false; !settled; ++level) {
      // Ensure we have enough depth.
      if (summary_levels_.size() <= level) {
        summary_levels_.emplace_back();
      }

      // Merge summaries.
      Summary& current_summary = summary_levels_[level];
      local_summary_.Merge(current_summary);

      // Check if we need to compress and propagate summary higher.
      if (current_summary.Size() == 0 ||
          local_summary_.Size() <= block_size_ + 1) {
        current_summary = std::move(local_summary_);
        settled = true;
      } else {
        // Compress, empty current level and propagate.
        local_summary_.Compress(block_size_, eps_);
        current_summary.Clear();
      }
    }
  }

  // Desired approximation precision.
  double eps_;
  // Maximum number of levels.
  int64 max_levels_;
  // Max block size per level.
  int64 block_size_;
  // Base buffer.
  Buffer buffer_;
  // Local summary used to minimize memory allocation and cache misses.
  // After the stream is finalized, this summary holds the final quantile
  // estimates.
  Summary local_summary_;
  // Summary levels;
  std::vector<Summary> summary_levels_;
  // Flag indicating whether the stream is finalized.
  bool finalized_;
};

template <typename ValueType, typename WeightType, typename CompareFn>
inline std::tuple<int64, int64>
WeightedQuantilesStream<ValueType, WeightType, CompareFn>::GetQuantileSpecs(
    double eps, int64 max_elements) {
  int64 max_level = 1LL;
  int64 block_size = 2LL;
  QCHECK(eps >= 0 && eps < 1);
  QCHECK_GT(max_elements, 0);

  if (eps <= std::numeric_limits<double>::epsilon()) {
    // Exact quantile computation at the expense of RAM.
    max_level = 1;
    block_size = std::max(max_elements, 2LL);
  } else {
    // The bottom-most level will become full at most
    // (max_elements / block_size) times, the level above will become full
    // (max_elements / 2 * block_size) times and generally level l becomes
    // full (max_elements / 2^l * block_size) times until the last
    // level max_level becomes full at most once meaning when the inequality
    // (2^max_level * block_size >= max_elements) is satisfied.
    // In what follows, we jointly solve for max_level and block_size by
    // gradually increasing the level until the inequality above is satisfied.
    // We could alternatively set max_level = ceil(log2(eps * max_elements));
    // and block_size = ceil(max_level / eps) + 1 but that tends to give more
    // pessimistic bounds and wastes RAM needlessly.
    for (max_level = 1, block_size = 2;
         (1LL << max_level) * block_size < max_elements; ++max_level) {
      // Update upper bound on block size at current level, we always
      // increase the estimate by 2 to hold the min/max elements seen so far.
      block_size = static_cast<size_t>(ceil(max_level / eps)) + 1;
    }
  }
  return std::make_tuple(max_level, std::max(block_size, 2LL));
}

}  // namespace quantiles
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_QUANTILES_WEIGHTED_QUANTILES_STREAM_H_
