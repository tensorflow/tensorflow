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
// FinishedNodes returns a 1-D tensor listing the nodes that are finished
// accumulating.
#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

using std::placeholders::_1;
using std::placeholders::_2;

using tensorforest::CheckTensorBounds;
using tensorforest::Sum;
using tensorforest::BestSplitDominatesClassificationBootstrap;
using tensorforest::BestSplitDominatesClassificationChebyshev;
using tensorforest::BestSplitDominatesClassificationHoeffding;
using tensorforest::BestSplitDominatesRegression;

namespace {

struct EvaluateParams {
  Tensor leaves;
  Tensor node_to_accumulator;
  Tensor accumulator_sums;
  Tensor birth_epochs;
  int current_epoch;
  int32 num_split_after_samples;
  int32 min_split_samples;
  int32 check_dominates_every_samples;
  bool need_random;
  int64 random_seed;
  std::function<bool(int, random::SimplePhilox*)> dominate_method;
};

void Evaluate(const EvaluateParams& params, mutex* mutex, int32 start,
              int32 end, std::unordered_set<int32>* final_finished_leaves,
              std::unordered_set<int32>* final_stale) {
  const auto leaves = params.leaves.unaligned_flat<int32>();
  const auto node_map = params.node_to_accumulator.unaligned_flat<int32>();
  const auto sums = params.accumulator_sums.tensor<float, 2>();
  const auto start_epochs = params.birth_epochs.unaligned_flat<int32>();

  const int32 num_accumulators =
      static_cast<int32>(params.accumulator_sums.shape().dim_size(0));

  std::vector<int32> finished_leaves;
  std::vector<int32> stale;

  std::unique_ptr<random::SimplePhilox> simple_philox;
  random::PhiloxRandom rnd_gen(params.random_seed);

  if (params.need_random) {
    simple_philox.reset(new random::SimplePhilox(&rnd_gen));
  }

  std::unordered_set<int32> visited;
  for (int32 i = start; i < end; i++) {
    const int32 leaf = internal::SubtleMustCopy(leaves(i));
    if (leaf == -1 || visited.find(leaf) != visited.end()) {
      continue;
    }
    if (!FastBoundsCheck(leaf, node_map.size())) {
      LOG(ERROR) << "leaf " << leaf << " not in valid range.";
    }
    const int32 accumulator = internal::SubtleMustCopy(node_map(leaf));
    if (accumulator < 0) {
      continue;
    }

    if (!FastBoundsCheck(accumulator, num_accumulators)) {
      LOG(ERROR) << "accumulator " << accumulator << " not in valid range.";
    }
    // The first column holds the number of samples seen.
    // For classification, this should be the sum of the other columns.
    int32 count = sums(accumulator, 0);

    if (params.current_epoch > start_epochs(leaf) + 1) {
      if (count >= params.min_split_samples) {
        finished_leaves.push_back(leaf);
      } else {
        stale.push_back(leaf);
      }
      continue;
    }

    if (count >= params.num_split_after_samples) {
      finished_leaves.push_back(leaf);
      continue;
    }

    if (count < params.min_split_samples) {
      continue;
    }

    if (count % params.check_dominates_every_samples != 0) {
      continue;
    }

    bool finished = params.dominate_method(accumulator, simple_philox.get());
    if (finished) {
      finished_leaves.push_back(leaf);
    }

    visited.insert(leaf);
  }
  mutex_lock m(*mutex);
  final_finished_leaves->insert(finished_leaves.begin(), finished_leaves.end());
  final_stale->insert(stale.begin(), stale.end());
}
}  // namespace


class FinishedNodes : public OpKernel {
 public:
  explicit FinishedNodes(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr(
        "regression", &regression_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "num_split_after_samples", &num_split_after_samples_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "min_split_samples", &min_split_samples_));
    OP_REQUIRES_OK(context, context->GetAttr(
        "dominate_fraction", &dominate_fraction_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("dominate_method", &dominate_method_));
    OP_REQUIRES_OK(context, context->GetAttr("random_seed", &random_seed_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("check_dominates_every_samples",
                                    &check_dominates_every_samples_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& leaf_tensor = context->input(0);
    const Tensor& node_to_accumulator = context->input(1);
    const Tensor& split_sums = context->input(2);
    const Tensor& split_squares = context->input(3);
    const Tensor& accumulator_sums = context->input(4);
    const Tensor& accumulator_squares = context->input(5);
    const Tensor& birth_epochs = context->input(6);
    const Tensor& current_epoch = context->input(7);

    OP_REQUIRES(context, leaf_tensor.shape().dims() == 1,
                errors::InvalidArgument(
                    "leaf_tensor should be one-dimensional"));
    OP_REQUIRES(context, node_to_accumulator.shape().dims() == 1,
                errors::InvalidArgument(
                    "node_to_accumulator should be one-dimensional"));
    OP_REQUIRES(context, split_sums.shape().dims() == 3,
                errors::InvalidArgument(
                    "split_sums should be three-dimensional"));
    OP_REQUIRES(context, accumulator_sums.shape().dims() == 2,
                errors::InvalidArgument(
                    "accumulator_sums should be two-dimensional"));
    OP_REQUIRES(context, birth_epochs.shape().dims() == 1,
                errors::InvalidArgument(
                    "birth_epochs should be one-dimensional"));
    OP_REQUIRES(
        context,
        birth_epochs.shape().dim_size(0) ==
        node_to_accumulator.shape().dim_size(0),
        errors::InvalidArgument(
            "birth_epochs and node_to_accumulator should be the same size."));

    // Check tensor bounds.
    if (!CheckTensorBounds(context, leaf_tensor)) return;
    if (!CheckTensorBounds(context, node_to_accumulator)) return;
    if (!CheckTensorBounds(context, split_sums)) return;
    if (!CheckTensorBounds(context, split_squares)) return;
    if (!CheckTensorBounds(context, accumulator_sums)) return;
    if (!CheckTensorBounds(context, accumulator_squares)) return;
    if (!CheckTensorBounds(context, birth_epochs)) return;
    if (!CheckTensorBounds(context, current_epoch)) return;

    const int32 epoch = current_epoch.unaligned_flat<int32>()(0);

    const int32 num_leaves = static_cast<int32>(
        leaf_tensor.shape().dim_size(0));

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    int num_threads = worker_threads->num_threads;

    EvaluateParams params;
    params.leaves = leaf_tensor;
    params.node_to_accumulator = node_to_accumulator;
    params.accumulator_sums = accumulator_sums;
    params.birth_epochs = birth_epochs;
    params.current_epoch = epoch;
    params.min_split_samples = min_split_samples_;
    params.num_split_after_samples = num_split_after_samples_;
    params.need_random = false;
    params.check_dominates_every_samples = check_dominates_every_samples_;

    if (regression_) {
      params.dominate_method =
          std::bind(&BestSplitDominatesRegression, accumulator_sums,
                    accumulator_squares, split_sums, split_squares, _1);
    } else {
      if (dominate_method_ == "none") {
        params.dominate_method = [](int, random::SimplePhilox*) {
          return false;
        };
      } else if (dominate_method_ == "hoeffding") {
        params.dominate_method =
            std::bind(&BestSplitDominatesClassificationHoeffding,
                      accumulator_sums, split_sums, _1, dominate_fraction_);
      } else if (dominate_method_ == "chebyshev") {
        params.dominate_method =
            std::bind(&BestSplitDominatesClassificationChebyshev,
                      accumulator_sums, split_sums, _1, dominate_fraction_);
      } else if (dominate_method_ == "bootstrap") {
        params.need_random = true;

        params.random_seed = random_seed_;
        if (params.random_seed == 0) {
          params.random_seed = static_cast<uint64>(Env::Default()->NowMicros());
        }

        params.dominate_method =
            std::bind(&BestSplitDominatesClassificationBootstrap,
                      accumulator_sums, split_sums, _1, dominate_fraction_, _2);
      } else {
        LOG(FATAL) << "Unknown dominate method " << dominate_method_;
      }
    }

    std::unordered_set<int32> finished_leaves;
    std::unordered_set<int32> stale;
    mutex m;
    // Require at least 100 leaves per thread.  I guess that's about 800 cost
    // per unit.  This isn't well defined.
    const int64 costPerUnit = 800;
    auto work = [&params, &finished_leaves, &stale, &m, num_leaves](int64 start,
                                                                    int64 end) {
      CHECK(start <= end);
      CHECK(end <= num_leaves);
      Evaluate(params, &m, static_cast<int32>(start), static_cast<int32>(end),
               &finished_leaves, &stale);
    };
    Shard(num_threads, worker_threads->workers, num_leaves, costPerUnit, work);

    // Copy to output.
    Tensor* output_finished = nullptr;
    TensorShape finished_shape;
    finished_shape.AddDim(finished_leaves.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, finished_shape,
                                            &output_finished));
    auto out_finished = output_finished->unaligned_flat<int32>();
    std::copy(finished_leaves.begin(), finished_leaves.end(),
              out_finished.data());

    Tensor* output_stale = nullptr;
    TensorShape stale_shape;
    stale_shape.AddDim(stale.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, stale_shape,
                                            &output_stale));
    auto out_stale = output_stale->unaligned_flat<int32>();
    std::copy(stale.begin(), stale.end(), out_stale.data());
  }

 private:
  bool regression_;
  int32 num_split_after_samples_;
  int32 min_split_samples_;
  float dominate_fraction_;
  string dominate_method_;
  int32 random_seed_;
  int32 check_dominates_every_samples_;
};

REGISTER_KERNEL_BUILDER(Name("FinishedNodes").Device(DEVICE_CPU),
                        FinishedNodes);

}  // namespace tensorflow
