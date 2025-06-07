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
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/proto/model_runtime_info.pb.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/culprit_finder/culprit_finder_utils.h"
#include "tensorflow/lite/tools/culprit_finder/interpreter_handler.h"
#include "tensorflow/lite/tools/culprit_finder/model_metadata_lib.h"
#include "tensorflow/lite/tools/culprit_finder/tflite_input_manager.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/model_loader.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tooling {
using ::tflite::Flag;
using ::tflite::Flags;
using ::tflite::Interpreter;
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
  TfLiteStatus RunCulpritFinder();

  TfLiteStatus RunCulpritFinderBinarySearch();
  int BinarySearchFindStartNode(int start_node, int end_node);
  int BinarySearchFindEndNode(int start_node, int end_node);

  TfLiteStatus RunCulpritFinderLinearSearch();

 private:
  tflite::tools::TfLiteDelegatePtr GetDelegate(int start_node, int end_node);
  TfLiteStatus PrepareCulpritFinder();
  bool CulpritSearchMatchCondition(OverallStat overall_stat);

  TfLiteStatus CalculateErrorStats(int start_node, int end_node,
                                   OverallStat* overall_stat,
                                   std::vector<int> intermediate_outputs = {});

  void MakeReport();
  TfLiteStatus NodeRangeAnalysis(int start_node, int end_node);
  inline void LogOverallStat(const OverallStat& overall_stat);

  std::unique_ptr<ModelMetadata> model_metadata_;
  std::unique_ptr<TfliteInputManager> input_manager_;
  std::unique_ptr<InterpreterHandler> interpreter_handler_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::Interpreter> interpreter_with_delegate_;
  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Contain delegate-related parameters that are initialized from
  // command-line flags.
  ToolParams params_;

  // A vector of <error_threshold, OverallStat> pairs.
  std::vector<std::pair<float, OverallStat>> overall_stats_;
};

}  // namespace tooling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_LIB_H_
