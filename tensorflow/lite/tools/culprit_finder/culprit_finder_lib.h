/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_LIB_H_
#define TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_LIB_H_

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/profiling/proto/model_runtime_info.pb.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/culprit_finder/culprit_finder_utils.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/model_loader.h"
#include "tensorflow/lite/tools/tool_params.h"
#include "tensorflow/lite/tools/utils.h"

namespace tflite {
namespace tooling {
using ::tflite::Flag;
using ::tflite::Flags;
using ::tflite::Interpreter;
using ::tflite::profiling::Edge;
using ::tflite::profiling::Node;
using ::tflite::tools::ModelLoader;
using ::tflite::tools::ProvidedDelegateList;
using ::tflite::tools::ToolParam;
using ::tflite::tools::ToolParams;

class CulpritFinder {
 public:
  CulpritFinder(int* argc, const char** argv);
  bool InitFromCmdlineArgs(int* argc, const char** argv);
  std::vector<Flag> GetFlags();
  void SetDefaultParams();
  void LogParams();

  std::string GetModelPath();
  absl::Status LoadModel();

  TfLiteStatus PrepareInterpreter(tflite::tools::TfLiteDelegatePtr delegate,
                                  std::vector<int> intermediate_outputs = {});
  tflite::tools::TfLiteDelegatePtr GetDelegate(int start_node, int end_node);

  TfLiteStatus PrepareInputData();
  TfLiteStatus SetInputTensors(Interpreter* interpreter);

  TfLiteStatus RunInference(Interpreter* interpreter);

  TfLiteStatus RunCulpritFinder();

  TfLiteStatus RunCulpritFinderLinearSearch();
  void MakeReport();

  TfLiteStatus RunCulpritFinderBinarySearch();
  int BinarySearchFindStartNode(int start_node, int end_node);
  int BinarySearchFindEndNode(int start_node, int end_node);
  TfLiteStatus NodeRangeAnalysis(int start_node, int end_node);

 private:
  TfLiteStatus PrepareCulpritFinder();
  bool CulpritSearchMatchCondition(OverallStat overall_stat);

  TfLiteStatus PopulateModelRuntimeDetails();

  // Helper functions for generating human readable reports.
  std::string EdgeShapeToString(const tflite::profiling::Edge& edge);
  std::string GetNodeIdentifier(int node_index, bool with_index = false);
  std::string GetTensorIdentifier(int tensor_index);
  std::string GetNodeShapes(int node_index);

  std::vector<int> GetOutputTensorsOfNode(int node_id);

  std::vector<int> GetNodeIdsInRange(int start_node, int end_node);

  TfLiteStatus CalculateErrorStats(int start_node, int end_node,
                                   OverallStat* overall_stat,
                                   std::vector<int> intermediate_outputs = {});

  inline void LogOverallStat(const OverallStat& overall_stat);

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<ModelLoader> model_loader_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::Interpreter> interpreter_with_delegate_;

  // Contain delegate-related parameters that are initialized from
  // command-line flags.
  ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;
  std::unordered_map<std::pair<int, int>, std::vector<OverallStat>,
                     PairHash<int, int>>
      tensor_stats_;
  std::vector<tflite::utils::InputTensorData> inputs_data_;
  std::vector<tflite::tools::TfLiteDelegatePtr> owned_delegates_;

  std::unordered_map<int, Node> node_index_to_node_proto_;
  std::unordered_map<int, Edge> tensor_index_to_edge_proto_;
  absl::flat_hash_map<int, std::vector<int>> tensor_index_to_dst_nodes_;
  absl::flat_hash_map<int, int> tensor_index_to_src_nodes_;
  std::vector<std::pair<float, OverallStat>> overall_stats_;
};

}  // namespace tooling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_LIB_H_
