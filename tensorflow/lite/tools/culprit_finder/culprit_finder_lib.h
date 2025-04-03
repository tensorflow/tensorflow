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

#include "absl/types/span.h"
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

  // Initialize the culprit finder from command line arguments.
  bool InitFromCmdlineArgs(int* argc, const char** argv);

  // Run the culprit finder.
  TfLiteStatus RunCulpritFinder();

  // Run the culprit finder binary search.
  TfLiteStatus RunCulpritFinderBinarySearch();
  // Find the start node of the culprit node range.
  int BinarySearchFindStartNode(int start_node, int end_node);
  // Find the end node of the culprit node range.
  int BinarySearchFindEndNode(int start_node, int end_node);

 private:
  // Get the delegate for the given node range. The delegate type and options
  // are determined by the flags passed in. The owner of the delegate is the
  // caller of this function. Only returns the first delegate if multiple
  // delegates are supported.
  tflite::tools::TfLiteDelegatePtr GetDelegate(int start_node, int end_node);

  // Prepare the culprit finder. This includes loading the model, creating the
  // reference interpreter, creating the input manager, and running an inference
  // on the reference interpreter.
  TfLiteStatus PrepareCulpritFinder();

  // Check if the overall stat matches the culprit search condition.
  bool CulpritSearchMatchCondition(const OverallStat& overall_stat);

  // Calculate the error stats for the given node range. If
  // intermediate_outputs is provided, the error stats will be calculated for
  // the intermediate outputs and the model outputs.
  // OverallStat is an output parameter that will be populated with the error
  // stats.
  TfLiteStatus CalculateErrorStats(int start_node, int end_node,
                                   absl::Span<const int> intermediate_outputs,
                                   OverallStat& overall_stat);

  // Calculate the error stats for the given node range.
  TfLiteStatus CalculateErrorStats(int start_node, int end_node,
                                   OverallStat& overall_stat) {
    return CalculateErrorStats(start_node, end_node, {}, overall_stat);
  }

  // Get the flags for the culprit finder.
  std::vector<Flag> GetFlags();

  // Set the default params for the culprit finder.
  void SetDefaultParams();

  // Log the params for the culprit finder.
  void LogParams();

  // Get the model path from the params.
  std::string GetModelPath();

  // The model metadata for the model.
  std::unique_ptr<ModelMetadata> model_metadata_;
  // The input manager for the model.
  std::unique_ptr<TfliteInputManager> input_manager_;
  // The interpreter handler for the model.
  std::unique_ptr<InterpreterHandler> interpreter_handler_;

  // The reference interpreter for the model.
  std::unique_ptr<tflite::Interpreter> interpreter_;
  // The interpreter with delegate for the model. This is used to calculate the
  // error stats for the delegated node range.
  std::unique_ptr<tflite::Interpreter> interpreter_with_delegate_;
  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Contain delegate-related parameters that are initialized from
  // command-line flags.
  ToolParams params_;
};

}  // namespace tooling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_CULPRIT_FINDER_CULPRIT_FINDER_LIB_H_
