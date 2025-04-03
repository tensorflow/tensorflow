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

#include "tensorflow/lite/tools/culprit_finder/culprit_finder_lib.h"

#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "third_party/odml/litert/litert/cc/litert_expected.h"
#include "third_party/odml/litert/litert/cc/litert_macros.h"
#include "third_party/odml/litert/litert/cc/litert_tflite_error_status_builder.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/culprit_finder/culprit_finder_utils.h"
#include "tensorflow/lite/tools/culprit_finder/interpreter_handler.h"
#include "tensorflow/lite/tools/culprit_finder/model_metadata_lib.h"
#include "tensorflow/lite/tools/culprit_finder/tflite_input_manager.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {

namespace tooling {

constexpr char kModelFileFlag[] = "graph";

// Binary search specific flags.
constexpr char kBinarySearchReverseSweepFlag[] = "binary_search_reverse_sweep";

// Find NAN specific flags.
constexpr char kFindNanFlag[] = "find_nan";
// Find numeric error specific flags.
constexpr char kFindNumericErrorFlag[] = "find_numeric_error";
constexpr char kMinNumericErrorFlag[] = "min_numeric_error";
// GPU specific flags.
constexpr char kGPUPrecisionLossAllowedFlag[] = "gpu_precision_loss_allowed";

using ::tflite::Flag;
using ::tflite::Flags;
using ::tflite::Interpreter;
using ::tflite::tools::ProvidedDelegateList;
using ::tflite::tools::ToolParam;

CulpritFinder::CulpritFinder(int* argc, const char** argv)
    : delegate_list_util_(&params_) {
  SetDefaultParams();
  bool parse_result = InitFromCmdlineArgs(argc, argv);

  if (!parse_result) {
    TFLITE_LOG(ERROR) << "Failed to parse command line arguments";
    return;
  }
  LogParams();

  if (GetModelPath().empty()) {
    TFLITE_LOG(ERROR) << "Model path is empty";
    return;
  }

  litert::Expected<std::unique_ptr<InterpreterHandler>> interpreter_handler =
      InterpreterHandler::Create(GetModelPath());
  if (!interpreter_handler) {
    TFLITE_LOG(ERROR) << "Failed to load model from path: " << GetModelPath();
    return;
  }
  interpreter_handler_ = std::move(*interpreter_handler);
  const std::vector<ProvidedDelegateList::ProvidedDelegate> delegates =
      delegate_list_util_.CreateAllRankedDelegates();
  if (delegates.size() != 1) {
    TFLITE_LOG(ERROR) << "Expected 1 delegate, got " << delegates.size();
    return;
  };
}

void CulpritFinder::SetDefaultParams() {
  params_.AddParam(kModelFileFlag, ToolParam::Create<std::string>(""));
  params_.AddParam(kBinarySearchReverseSweepFlag,
                   ToolParam::Create<bool>(false));
  params_.AddParam(kFindNanFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kFindNumericErrorFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kMinNumericErrorFlag, ToolParam::Create<float>(0.0001));
  params_.AddParam(kGPUPrecisionLossAllowedFlag, ToolParam::Create<bool>(true));
  delegate_list_util_.AddAllDelegateParams();
}

void CulpritFinder::LogParams() {
  LOG_TOOL_PARAM(params_, std::string, kModelFileFlag, "Model file", true);
  LOG_TOOL_PARAM(params_, bool, kBinarySearchReverseSweepFlag,
                 "Binary search find end first", true);
  LOG_TOOL_PARAM(params_, bool, kFindNanFlag, "Find NAN", true);
  LOG_TOOL_PARAM(params_, bool, kFindNumericErrorFlag, "Find numeric error",
                 true);
  LOG_TOOL_PARAM(params_, float, kMinNumericErrorFlag, "Min numeric error",
                 true);
  LOG_TOOL_PARAM(params_, bool, kGPUPrecisionLossAllowedFlag,
                 "Allow GPU precision loss", true);
  for (const std::unique_ptr<tflite::tools::DelegateProvider>&
           delegate_provider :
       tflite::tools::GetRegisteredDelegateProviders()) {
    delegate_provider->LogParams(params_, true);
  }
}

std::vector<tflite::Flag> CulpritFinder::GetFlags() {
  std::vector<tflite::Flag> flag_list = {
      CreateFlag<std::string>(kModelFileFlag, &params_,
                              "Path to test tflite model file."),
      CreateFlag<bool>(kBinarySearchReverseSweepFlag, &params_,
                       "If true, find the end node first. Default is false."),
      CreateFlag<bool>(kFindNanFlag, &params_,
                       "If specified, searches for NANs."),
      CreateFlag<bool>(kFindNumericErrorFlag, &params_,
                       "If specified, searches for numeric errors."),
      CreateFlag<float>(kMinNumericErrorFlag, &params_,
                        "Minimum absolute difference to consider an "
                        "inference as an error."),
      CreateFlag<bool>(kGPUPrecisionLossAllowedFlag, &params_,
                       "Allow GPU precision loss."),
  };
  delegate_list_util_.AppendCmdlineFlags(flag_list);
  return flag_list;
}

bool CulpritFinder::InitFromCmdlineArgs(int* argc, const char** argv) {
  const std::vector<Flag> flags = GetFlags();
  bool parse_result = Flags::Parse(argc, argv, flags);
  if (!parse_result || params_.Get<bool>("help")) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
    // Returning false intentionally when "--help=true" is specified so that
    // the caller could check the return value to decide stopping the
    // execution.
    parse_result = false;
  }
  return parse_result;
}

std::string CulpritFinder::GetModelPath() {
  return params_.Get<std::string>(kModelFileFlag);
}

tflite::tools::TfLiteDelegatePtr CulpritFinder::GetDelegate(int start_node,
                                                            int end_node) {
  params_.Set<int>("first_delegate_node_index", start_node, 0);
  params_.Set<int>("last_delegate_node_index", end_node, 0);
  std::vector<ProvidedDelegateList::ProvidedDelegate> delegates =
      delegate_list_util_.CreateAllRankedDelegates();
  if (delegates.empty()) {
    return tflite::tools::CreateNullDelegate();
  }
  return std::move(delegates[0].delegate);
}

TfLiteStatus CulpritFinder::CalculateErrorStats(
    const int start_node, const int end_node,
    absl::Span<const int> intermediate_outputs, OverallStat& overall_stat) {
  bool is_crash = false;
  try {
    LITERT_ASSIGN_OR_RETURN(
        interpreter_with_delegate_,
        interpreter_handler_->PrepareInterpreter(
            GetDelegate(start_node, end_node), intermediate_outputs),
        AsTfLiteStatus(_ << "Failed to prepare interpreter."));

    TfLiteStatus status = interpreter_handler_->RunInference(
        *interpreter_with_delegate_, *input_manager_);
    if (status != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to run inference";
      return kTfLiteError;
    }
  } catch (const std::exception& exc) {
    TFLITE_LOG(ERROR) << "Failed to run inference due to a crash.";
    is_crash = true;
  }

  GetOverallStat(start_node, end_node, interpreter_.get(),
                 interpreter_with_delegate_.get(), is_crash, overall_stat);
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::RunCulpritFinder() {
  std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor>
      peak_memory_reporter;
  peak_memory_reporter =
      std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(50);
  peak_memory_reporter->Start();
  TfLiteStatus status = RunCulpritFinderBinarySearch();
  peak_memory_reporter->Stop();
  TFLITE_LOG(INFO) << "### Peak memory usage in MB: "
                   << peak_memory_reporter->GetPeakMemUsageInMB();
  return status;
}

TfLiteStatus CulpritFinder::PrepareCulpritFinder() {
  LITERT_ASSIGN_OR_RETURN(
      interpreter_,
      interpreter_handler_->PrepareInterpreter(
          tflite::tools::CreateNullDelegate()),
      AsTfLiteStatus(_ << "Failed to prepare interpreter."));

  TFLITE_LOG(INFO) << "Reference interpreter prepared";

  LITERT_ASSIGN_OR_RETURN(
      model_metadata_,
      tflite::tooling::ModelMetadata::Create(interpreter_.get()),
      AsTfLiteStatus(_ << "Failed to create model info."));

  input_manager_ = std::make_unique<TfliteInputManager>(interpreter_.get());
  if (input_manager_->PrepareInputData() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare input data";
    return kTfLiteError;
  }
  if (interpreter_handler_->RunInference(*interpreter_, *input_manager_) !=
      kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to run reference inference";
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Reference inference run completed!";
  return kTfLiteOk;
}

bool CulpritFinder::CulpritSearchMatchCondition(
    const OverallStat& overall_stat) {
  if (params_.Get<bool>(kFindNanFlag) &&
      !overall_stat.nan_output_indices.empty()) {
    return true;
  }
  if (params_.Get<bool>(kFindNumericErrorFlag) &&
      overall_stat.total_error >= params_.Get<float>(kMinNumericErrorFlag)) {
    return true;
  }
  if (overall_stat.is_crash) {
    return true;
  }
  return false;
}

TfLiteStatus CulpritFinder::RunCulpritFinderBinarySearch() {
  if (PrepareCulpritFinder() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare culprit finder";
    return kTfLiteError;
  }
  int start_node = 0;
  int end_node = interpreter_->nodes_size();

  OverallStat temp_overall_stat;
  if (CalculateErrorStats(start_node, end_node, temp_overall_stat) !=
      kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to calculate error stats";
    return kTfLiteError;
  } else if (!CulpritSearchMatchCondition(temp_overall_stat)) {
    TFLITE_LOG(INFO) << "No nan outputs/numeric errors found";
    return kTfLiteOk;
  }

  if (params_.Get<bool>(kBinarySearchReverseSweepFlag)) {
    end_node = BinarySearchFindEndNode(start_node, end_node);
    TFLITE_LOG(INFO) << "### Found min end_node: " << end_node;
    start_node = BinarySearchFindStartNode(start_node, end_node);
  } else {
    start_node = BinarySearchFindStartNode(start_node, end_node);
    TFLITE_LOG(INFO) << "### Found max start_node: " << start_node;
    end_node = BinarySearchFindEndNode(start_node, end_node);
  }

  TFLITE_LOG(INFO) << "### Culprit node range: [" << start_node << " - "
                   << end_node << "]";

  return kTfLiteOk;
}

int CulpritFinder::BinarySearchFindStartNode(int start_node, int end_node) {
  int start_node_range_start = start_node;
  int start_node_range_end = end_node;
  while (start_node_range_start <= start_node_range_end) {
    OverallStat overall_stat;
    const int mid_node =
        std::floor((start_node_range_start + start_node_range_end) / 2);
    TFLITE_LOG(INFO) << "Looking for start node in node range: ["
                     << start_node_range_start << " - " << start_node_range_end
                     << "] by computing error stats for range [" << mid_node
                     << " - " << end_node << "]";
    const std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(end_node);
    if (CalculateErrorStats(mid_node, end_node, output_tensors, overall_stat) !=
        kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to calculate error stats";
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      start_node = mid_node;
      start_node_range_start = mid_node + 1;
    } else {
      start_node_range_end = mid_node - 1;
    }
  }
  return start_node;
}

int CulpritFinder::BinarySearchFindEndNode(int start_node, int end_node) {
  int end_node_range_start = start_node;
  int end_node_range_end = end_node;

  while (end_node_range_start <= end_node_range_end) {
    OverallStat overall_stat;
    const int mid_node =
        std::floor((end_node_range_start + end_node_range_end) / 2);
    TFLITE_LOG(INFO) << "Looking for end node in node range: ["
                     << end_node_range_start << " - " << end_node_range_end
                     << "] by computing error stats for range [" << start_node
                     << " - " << mid_node << "]";
    const std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(mid_node);
    if (CalculateErrorStats(start_node, mid_node, output_tensors,
                            overall_stat) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to calculate error stats";
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      end_node = mid_node;
      end_node_range_end = mid_node - 1;
    } else {
      end_node_range_start = mid_node + 1;
    }
  }
  return end_node;
}

}  // namespace tooling
}  // namespace tflite
