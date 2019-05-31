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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GRAPH_COLLECTION_OPERATOR_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GRAPH_COLLECTION_OPERATOR_H_

#include <vector>
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/candidate_graph_runner.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/grow_stats.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/split_collection_operators.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"

namespace tensorflow {
namespace tensorforest {

// Holds split candidates that are trained by running any TF graph.
class GraphRunnerSplitCollectionOperator : public SplitCollectionOperator {
 public:
  explicit GraphRunnerSplitCollectionOperator(const TensorForestParams& params)
      : SplitCollectionOperator(params) {
    if (params.num_splits_to_consider().ParamType_case() ==
        DepthDependentParam::PARAMTYPE_NOT_SET) {
      LOG(FATAL) << "GRAPH_RUNNER_COLLECTION must specify a constant value for "
                 << " num_splits_to_consider";
    } else {
      num_splits_to_consider_ =
          params.num_splits_to_consider().constant_value();
    }
  }

  std::unique_ptr<GrowStats> CreateGrowStats(int32 node_id,
                                             int32 depth) const override;

  // Updates the slot's candidates with the new example.
  // Assumes slot has been initialized.
  void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                  const InputTarget* target, const std::vector<int>& examples,
                  int32 node_id) const override;

  // Create a new candidate and initialize it with the given example.
  void CreateAndInitializeCandidateWithExample(
      const std::unique_ptr<TensorDataSet>& input_data,
      const InputTarget* target, int example, int32 node_id) const override;

  bool BestSplit(int32 node_id, SplitCandidate* best,
                 int32* depth) const override;

  void ClearSlot(int32 node_id) override;

 protected:
  int64 UniqueId(int32 node_id, int32 split_id) const;

  mutable std::unordered_map<int64, std::unique_ptr<CandidateGraphRunner>>
      runners_;
  int features_per_node_;
  string graph_dir_;
  // Must have a constant value because of how we make unique ids right now.
  int32 num_splits_to_consider_;
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GRAPH_COLLECTION_OPERATOR_H_
