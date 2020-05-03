// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_QUANTILES_QUANTILE_STREAM_RESOURCE_H_
#define TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_QUANTILES_QUANTILE_STREAM_RESOURCE_H_

#include <vector>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/boosted_trees/quantiles/weighted_quantiles_stream.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

using QuantileStream =
    boosted_trees::quantiles::WeightedQuantilesStream<float, float>;

// Quantile Stream Resource for a list of streams sharing the same number of
// quantiles, maximum elements, and epsilon.
class BoostedTreesQuantileStreamResource : public ResourceBase {
 public:
  BoostedTreesQuantileStreamResource(const float epsilon,
                                     const int64 max_elements,
                                     const int64 num_streams)
      : are_buckets_ready_(false),
        epsilon_(epsilon),
        num_streams_(num_streams),
        max_elements_(max_elements) {
    streams_.reserve(num_streams_);
    boundaries_.reserve(num_streams_);
    for (int64 idx = 0; idx < num_streams; ++idx) {
      streams_.push_back(QuantileStream(epsilon, max_elements));
      boundaries_.push_back(std::vector<float>());
    }
  }

  string DebugString() const override { return "QuantileStreamResource"; }

  tensorflow::mutex* mutex() { return &mu_; }

  QuantileStream* stream(const int64 index) { return &streams_[index]; }

  const std::vector<float>& boundaries(const int64 index) {
    return boundaries_[index];
  }

  void set_boundaries(const std::vector<float>& boundaries, const int64 index) {
    boundaries_[index] = boundaries;
  }

  float epsilon() const { return epsilon_; }
  int64 num_streams() const { return num_streams_; }

  bool are_buckets_ready() const { return are_buckets_ready_; }
  void set_buckets_ready(const bool are_buckets_ready) {
    are_buckets_ready_ = are_buckets_ready;
  }

  void ResetStreams() {
    streams_.clear();
    streams_.reserve(num_streams_);
    for (int64 idx = 0; idx < num_streams_; ++idx) {
      streams_.push_back(QuantileStream(epsilon_, max_elements_));
    }
  }

 private:
  ~BoostedTreesQuantileStreamResource() override {}

  // Mutex for the whole resource.
  tensorflow::mutex mu_;

  // Quantile streams.
  std::vector<QuantileStream> streams_;

  // Stores the boundaries. Same size as streams_.
  std::vector<std::vector<float>> boundaries_;

  // Whether boundaries are created. Initially boundaries are empty until
  // set_boundaries are called.
  bool are_buckets_ready_;

  const float epsilon_;
  const int64 num_streams_;
  // An upper-bound for the number of elements.
  int64 max_elements_;

  TF_DISALLOW_COPY_AND_ASSIGN(BoostedTreesQuantileStreamResource);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_QUANTILES_QUANTILE_STREAM_RESOURCE_H_
