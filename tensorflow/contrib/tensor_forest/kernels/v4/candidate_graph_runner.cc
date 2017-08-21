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
#include "tensorflow/contrib/tensor_forest/kernels/v4/candidate_graph_runner.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace tensorforest {

// Names of ops in the graph to run.
constexpr char kInitializeOp[] = "init";
constexpr char kAddExampleOp[] = "add_example";
constexpr char kSplitScoreName[] = "split_score";
constexpr char kGetSplitName[] = "get_split";
constexpr char kGetLeftStatsName[] = "get_left_stats";
constexpr char kGetRightStatsName[] = "get_right_stats";

// Names of files written by python graph builder.
constexpr char kGraphFilename[] = "graph";
constexpr char kSaverDefFilename[] = "saver";
constexpr char kMetaDefFilename[] = "meta";

// Names of Tensor inputs.
constexpr char kFeaturesName[] = "features";
constexpr char kInputDataName[] = "input_data";
constexpr char kTargetsName[] = "targets";
constexpr char kExamplesName[] = "examples";

constexpr char kNoOp[] = "none";

CandidateGraphRunner::CandidateGraphRunner(
    const string& graph_dir, const decision_trees::BinaryNode& split)
    : split_(split) {
  // read graph from file.
  GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(
      Env::Default(), io::JoinPath(graph_dir, kGraphFilename), &graph_def))
      << "Could not read graph def.";

  // create session.
  session_.reset(::tensorflow::NewSession(SessionOptions()));
  TF_CHECK_OK(session_->Create(graph_def)) << "Failed to create session";

  // Features don't change, store them in a tensor.
  const auto& oblique = split.inequality_left_child_test().oblique();
  const int32 feat_size = oblique.features_size();
  features_.reset(
      new Tensor(tensorflow::DT_INT32, TensorShape({feat_size})));
  auto feat = features_->flat<int32>();
  int i = 0;
  for (const auto& id : oblique.features()) {
    safe_strto32(id.id().value(), &feat(i++));
  }
}

void CandidateGraphRunner::RunOp(
    const string& name, const TensorNameValueList& inputs,
    const std::vector<string>& output_tensor_names,
    std::vector<Tensor>* outputs) {
  std::vector<string> op_name;
  if (name != kNoOp) {
    op_name.push_back(name);
  }
  TF_CHECK_OK(session_->Run(inputs, output_tensor_names, op_name, outputs))
      << "Failed to run: " << name;
}

void CandidateGraphRunner::Init() {
  RunOp(kInitializeOp, TensorNameValueList(), std::vector<string>(), nullptr);
}

void CandidateGraphRunner::AddExample(const Tensor& input_data,
                                      const Tensor& target,
                                      const Tensor& examples) {
  TensorNameValueList inputs;
  inputs.emplace_back(kFeaturesName, *features_);
  inputs.emplace_back(kExamplesName, examples);
  inputs.emplace_back(kInputDataName, input_data);
  inputs.emplace_back(kTargetsName, target);

  RunOp(kAddExampleOp, inputs, std::vector<string>(), nullptr);
}

float CandidateGraphRunner::SplitScore() {
  std::vector<Tensor> outputs;
  RunOp(kNoOp, TensorNameValueList(), {kSplitScoreName}, &outputs);
  return outputs[0].unaligned_flat<float>()(0);
}

void CandidateGraphRunner::GetSplit(decision_trees::BinaryNode* node) {
  std::vector<Tensor> outputs;
  RunOp(kNoOp, TensorNameValueList(), {kGetSplitName}, &outputs);
  ParseProtoUnlimited(node, outputs[0].unaligned_flat<string>()(0));
  const auto& oblique = split_.inequality_left_child_test().oblique();
  auto* new_split =
      node->mutable_inequality_left_child_test()->mutable_oblique();
  for (const auto& id : oblique.features()) {
    *new_split->add_features() = id;
  }
}

void CandidateGraphRunner::GetLeftStats(LeafStat* stats) {
  std::vector<Tensor> outputs;
  RunOp(kNoOp, TensorNameValueList(), {kGetLeftStatsName}, &outputs);
  const auto& counts = outputs[0].unaligned_flat<float>();
  auto* dense = stats->mutable_classification()->mutable_dense_counts();
  for (int i = 0; i < counts.size(); ++i) {
    dense->add_value()->set_float_value(counts(i));
  }
}

void CandidateGraphRunner::GetRightStats(LeafStat* stats) {
  std::vector<Tensor> outputs;
  RunOp(kNoOp, TensorNameValueList(), {kGetRightStatsName}, &outputs);
  const auto& counts = outputs[0].unaligned_flat<float>();
  auto* dense = stats->mutable_classification()->mutable_dense_counts();
  for (int i = 0; i < counts.size(); ++i) {
    dense->add_value()->set_float_value(counts(i));
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
