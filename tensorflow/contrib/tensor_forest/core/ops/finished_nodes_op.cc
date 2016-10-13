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
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

using tensorforest::CheckTensorBounds;
using tensorforest::Sum;
using tensorforest::BestSplitDominatesClassification;
using tensorforest::BestSplitDominatesRegression;

namespace {

struct EvaluateParams {
  Tensor leaves;
  Tensor node_to_accumulator;
  Tensor split_sums;
  Tensor split_squares;
  Tensor accumulator_sums;
  Tensor accumulator_squares;
  Tensor birth_epochs;
  int current_epoch;
  float dominate_fraction;
  int32 num_split_after_samples;
  int32 min_split_samples;
  bool regression;
};

void Evaluate(const EvaluateParams& params, mutex* mutex, int32 start,
              int32 end, std::vector<int32>* final_finished_leaves,
              std::vector<int32>* final_stale) {
  const auto leaves = params.leaves.unaligned_flat<int32>();
  const auto node_map = params.node_to_accumulator.unaligned_flat<int32>();
  const auto sums = params.accumulator_sums.tensor<float, 2>();
  const auto start_epochs = params.birth_epochs.unaligned_flat<int32>();

  const int32 num_accumulators =
      static_cast<int32>(params.accumulator_sums.shape().dim_size(0));

  std::vector<int32> finished_leaves;
  std::vector<int32> stale;

  for (int32 i = start; i < end; i++) {
    const int32 leaf = internal::SubtleMustCopy(leaves(i));
    if (leaf == -1) {
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

    bool finished = false;
    if (params.regression) {
      finished = BestSplitDominatesRegression(
          params.accumulator_sums, params.accumulator_squares,
          params.split_sums, params.split_squares, accumulator);
    } else {
      finished = BestSplitDominatesClassification(
          params.accumulator_sums, params.split_sums, accumulator,
          params.dominate_fraction);
    }

    if (finished) {
      finished_leaves.push_back(leaf);
    }
  }
  mutex_lock m(*mutex);
  final_finished_leaves->insert(final_finished_leaves->end(),
                                finished_leaves.begin(), finished_leaves.end());
  final_stale->insert(final_stale->end(), stale.begin(), stale.end());
}
}  // namespace

REGISTER_OP("FinishedNodes")
    .Attr("regression: bool = false")
    .Attr("num_split_after_samples: int")
    .Attr("min_split_samples: int")
    .Attr("dominate_fraction: float = 0.99")
    .Input("leaves: int32")
    .Input("node_to_accumulator: int32")
    .Input("split_sums: float")
    .Input("split_squares: float")
    .Input("accumulator_sums: float")
    .Input("accumulator_squares: float")
    .Input("birth_epochs: int32")
    .Input("current_epoch: int32")
    .Output("finished: int32")
    .Output("stale: int32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
Determines which of the given leaf nodes are done accumulating.

leaves:= A 1-d int32 tensor.  Lists the nodes that are currently leaves.
node_to_accumulator: If the i-th node is fertile, `node_to_accumulator[i]`
  is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
split_sums:= a 3-d tensor where `split_sums[a][s]` summarizes the
  training labels for examples that fall into the fertile node associated with
  accumulator slot s and have then taken the *left* branch of candidate split
  s.  For a classification problem, `split_sums[a][s][c]` is the count of such
  examples with class c and for regression problems, `split_sums[a][s]` is the
  sum of the regression labels for such examples.
split_squares: Same as split_sums, but it contains the sum of the
  squares of the regression labels.  Only used for regression.  For
  classification problems, pass a dummy tensor into this.
accumulator_sums: For classification, `accumulator_sums[a][c]` records how
  many training examples have class c and have ended up in the fertile node
  associated with accumulator slot a.  It has the total sum in entry 0 for
  convenience. For regression, it is the same except it contains the sum
  of the input labels that have been seen, and entry 0 contains the number
  of training examples that have been seen.
accumulator_squares: Same as accumulator_sums, but it contains the sum of the
  squares of the regression labels.  Only used for regression.  For
  classification problems, pass a dummy tensor into this.
birth_epochs:= A 1-d int32 tensor.  `birth_epochs[i]` contains the epoch
  the i-th node was created in.
current_epoch:= A 1-d int32 tensor with shape (1).  `current_epoch[0]`
  stores the current epoch number.
finished:= A 1-d int32 tensor containing the indices of the finished nodes.
  Nodes are finished if they have received at least num_split_after_samples
  samples, or if they have received min_split_samples and the best scoring
  split is sufficiently greater than the next best split.
stale:= A 1-d int32 tensor containing the fertile nodes that were created two
  or more epochs ago.

)doc");

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
    params.split_sums = split_sums;
    params.split_squares = split_squares;
    params.accumulator_sums = accumulator_sums;
    params.birth_epochs = birth_epochs;
    params.current_epoch = epoch;
    params.dominate_fraction = dominate_fraction_;
    params.min_split_samples = min_split_samples_;
    params.num_split_after_samples = num_split_after_samples_;
    params.regression = regression_;

    std::vector<int32> finished_leaves;
    std::vector<int32> stale;
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
};

REGISTER_KERNEL_BUILDER(Name("FinishedNodes").Device(DEVICE_CPU),
                        FinishedNodes);

}  // namespace tensorflow
