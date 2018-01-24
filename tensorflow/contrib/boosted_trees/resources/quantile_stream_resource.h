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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_QUANTILE_STREAM_RESOURCE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_QUANTILE_STREAM_RESOURCE_H_

#include "tensorflow/contrib/boosted_trees/lib/quantiles/weighted_quantiles_stream.h"
#include "tensorflow/contrib/boosted_trees/proto/quantiles.pb.h"  // NOLINT
#include "tensorflow/contrib/boosted_trees/resources/stamped_resource.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace boosted_trees {

using QuantileStream =
    boosted_trees::quantiles::WeightedQuantilesStream<float, float>;

// Resource for accumulating summaries for multiple columns.
class QuantileStreamResource : public StampedResource {
 public:
  QuantileStreamResource(const float epsilon, const int32 num_quantiles,
                         const int64 max_elements, bool generate_quantiles,
                         int64 stamp_token)
      : stream_(epsilon, max_elements),
        are_buckets_ready_(false),
        epsilon_(epsilon),
        num_quantiles_(num_quantiles),
        max_elements_(max_elements),
        generate_quantiles_(generate_quantiles) {
    set_stamp(stamp_token);
  }

  string DebugString() override { return "QuantileStreamResource"; }

  tensorflow::mutex* mutex() { return &mu_; }

  QuantileStream* stream(int64 stamp) {
    CHECK(is_stamp_valid(stamp));
    return &stream_;
  }

  const std::vector<float>& boundaries(int64 stamp) {
    CHECK(is_stamp_valid(stamp));
    return boundaries_;
  }

  void set_boundaries(int64 stamp, const std::vector<float>& boundaries) {
    CHECK(is_stamp_valid(stamp));
    are_buckets_ready_ = true;
    boundaries_ = boundaries;
  }

  float epsilon() const { return epsilon_; }
  int32 num_quantiles() const { return num_quantiles_; }

  void Reset(int64 stamp) {
    set_stamp(stamp);
    stream_ = QuantileStream(epsilon_, max_elements_);
  }

  bool are_buckets_ready() const { return are_buckets_ready_; }
  void set_buckets_ready(bool are_buckets_ready) {
    are_buckets_ready_ = are_buckets_ready;
  }

  bool generate_quantiles() const { return generate_quantiles_; }
  void set_generate_quantiles(bool generate_quantiles) {
    generate_quantiles_ = generate_quantiles;
  }

 private:
  ~QuantileStreamResource() override {}

  // Mutex for the whole resource.
  tensorflow::mutex mu_;

  // Quantile stream.
  QuantileStream stream_;

  // Stores the boundaries from the previous iteration. Empty during the first
  // iteration.
  std::vector<float> boundaries_;

  // Whether boundaries are created. Initially boundaries are empty until
  // set_boundaries are called.
  bool are_buckets_ready_;

  const float epsilon_;
  const int32 num_quantiles_;
  // An upper-bound for the number of elements.
  int64 max_elements_;

  // Generate quantiles instead of approximate boundaries.
  // If true, exactly `num_quantiles` will be produced in the final summary.
  bool generate_quantiles_;

  TF_DISALLOW_COPY_AND_ASSIGN(QuantileStreamResource);
};

}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_QUANTILE_STREAM_RESOURCE_H_
