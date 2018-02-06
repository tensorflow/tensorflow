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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_CANDIDATE_GRAPH_RUNNER_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_CANDIDATE_GRAPH_RUNNER_H_
#include <string>
#include <vector>

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace tensorforest {

typedef std::vector<std::pair<string, ::tensorflow::Tensor>>
    TensorNameValueList;

// Class that represents one split candidate, and can perform operations
// on a session created from a graph.
class CandidateGraphRunner {
 public:
  // split should contain the features that are being used.
  CandidateGraphRunner(const string& graph_dir,
                       const decision_trees::BinaryNode& split);

  // Input the given data and target Tensors to the add_example op.
  void AddExample(const Tensor& input_data, const Tensor& target,
                  const Tensor& examples);

  // Get the candidates' split score with the split_score op.
  float SplitScore();

  // Fills in the split in node with weights and threshold.
  void GetSplit(decision_trees::BinaryNode* node);

  // Fills in the stats for the left-branch taken.
  void GetLeftStats(LeafStat* stats);

  // Fills in the stats for the right-branch taken.
  void GetRightStats(LeafStat* stats);

  // Initializes variables, must be run before other ops.
  void Init();

 protected:
  void RunOp(const string& name, const TensorNameValueList& inputs,
             const std::vector<string>& output_tensor_names,
             std::vector<Tensor>* outputs);

  std::unique_ptr<Session> session_;
  decision_trees::BinaryNode split_;
  std::unique_ptr<Tensor> features_;
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_CANDIDATE_GRAPH_RUNNER_H_
