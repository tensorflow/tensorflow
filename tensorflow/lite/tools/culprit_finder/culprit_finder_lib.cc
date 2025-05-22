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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"
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

constexpr char kModelFileFlag[] = "model_file";
constexpr char kSearchStrategyFlag[] = "search_strategy";

// Search strategy enums.
constexpr char kBinarySearchStrategyEnum[] = "binary";
constexpr char kLinearSearchStrategyEnum[] = "linear";

// Binary search specific flags.
constexpr char kBinarySearchReverseSweepFlag[] = "binary_search_reverse_sweep";

// Linear search specific flags.
constexpr char kLinearSearchBatchSizeFlag[] = "linear_search_batch_size";
constexpr char kLinearSearchNodeFilterFlag[] = "linear_search_node_filter";

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
  std::optional<std::unique_ptr<InterpreterHandler>> interpreter_handler =
      InterpreterHandler::Create(GetModelPath());
  if (interpreter_handler == std::nullopt) {
    TFLITE_LOG(ERROR) << "Failed to load model from path: " << GetModelPath();
    return;
  }
  interpreter_handler_ = std::move(*interpreter_handler);
  auto delegates = delegate_list_util_.CreateAllRankedDelegates();
  if (delegates.size() != 1) {
    TFLITE_LOG(ERROR) << "Expected 1 delegate, got " << delegates.size();
    return;
  };
}

void CulpritFinder::SetDefaultParams() {
  params_.AddParam(kModelFileFlag, ToolParam::Create<std::string>(""));
  params_.AddParam(kSearchStrategyFlag,
                   ToolParam::Create<std::string>(kLinearSearchStrategyEnum));
  params_.AddParam(kLinearSearchBatchSizeFlag, ToolParam::Create<int>(1));
  params_.AddParam(kLinearSearchNodeFilterFlag,
                   ToolParam::Create<std::string>(""));
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
  LOG_TOOL_PARAM(params_, std::string, kSearchStrategyFlag, "Search strategy",
                 true);
  LOG_TOOL_PARAM(params_, int, kLinearSearchBatchSizeFlag,
                 "Linear search batch size", true);
  LOG_TOOL_PARAM(params_, std::string, kLinearSearchNodeFilterFlag,
                 "Linear search node filter", true);
  LOG_TOOL_PARAM(params_, bool, kBinarySearchReverseSweepFlag,
                 "Binary search find end first", true);
  LOG_TOOL_PARAM(params_, bool, kFindNanFlag, "Find NAN", true);
  LOG_TOOL_PARAM(params_, bool, kFindNumericErrorFlag, "Find numeric error",
                 true);
  LOG_TOOL_PARAM(params_, float, kMinNumericErrorFlag, "Min numeric error",
                 true);
  LOG_TOOL_PARAM(params_, bool, kGPUPrecisionLossAllowedFlag,
                 "Allow GPU precision loss", true);
  for (const auto& delegate_provider :
       tflite::tools::GetRegisteredDelegateProviders()) {
    delegate_provider->LogParams(params_, true);
  }
}

std::vector<tflite::Flag> CulpritFinder::GetFlags() {
  std::vector<tflite::Flag> flag_list = {
      CreateFlag<std::string>(kModelFileFlag, &params_,
                              "Path to test tflite model file."),
      CreateFlag<std::string>(kSearchStrategyFlag, &params_,
                              "Search strategy (binary or linear)."),
      CreateFlag<int>(kLinearSearchBatchSizeFlag, &params_,
                      "If provided, the culprit finder will run the linear "
                      "search for batches of this size."),
      CreateFlag<std::string>(
          kLinearSearchNodeFilterFlag, &params_,
          "A comma-separated list of node types to filter out."),
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
  std::vector<Flag> flags = GetFlags();
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
  auto delegates = delegate_list_util_.CreateAllRankedDelegates();
  return std::move(delegates[0].delegate);
}

TfLiteStatus CulpritFinder::CalculateErrorStats(
    int start_node, int end_node, OverallStat* overall_stat,
    std::vector<int> intermediate_outputs) {
  bool is_crash = false;
  try {
    std::optional<std::unique_ptr<tflite::Interpreter>> interpreter =
        interpreter_handler_->PrepareInterpreter(
            GetDelegate(start_node, end_node), intermediate_outputs);
    if (interpreter == std::nullopt) {
      TFLITE_LOG(ERROR) << "Failed to prepare interpreter";
      return kTfLiteError;
    }
    interpreter_with_delegate_ = std::move(*interpreter);

    auto status = interpreter_handler_->RunInference(
        interpreter_with_delegate_.get(), input_manager_.get());
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
  std::string search_strategy;
  if (params_.HasValueSet<std::string>(kSearchStrategyFlag)) {
    search_strategy = params_.Get<std::string>(kSearchStrategyFlag);
  } else if (params_.Get<bool>(kFindNanFlag)) {
    search_strategy = kBinarySearchStrategyEnum;
  } else {
    search_strategy = kLinearSearchStrategyEnum;
  }
  auto status = kTfLiteOk;
  if (search_strategy == kLinearSearchStrategyEnum) {
    if (RunCulpritFinderLinearSearch() != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to run culprit finder linear search";
      status = kTfLiteError;
    }
  } else if (search_strategy == kBinarySearchStrategyEnum) {
    status = RunCulpritFinderBinarySearch();
  } else {
    TFLITE_LOG(ERROR) << "Unsupported search strategy: " << search_strategy;
    status = kTfLiteError;
  }
  peak_memory_reporter->Stop();
  TFLITE_LOG(INFO) << "### Peak memory usage in MB: "
                   << peak_memory_reporter->GetPeakMemUsageInMB();
  return status;
}

TfLiteStatus CulpritFinder::PrepareCulpritFinder() {
  std::optional<std::unique_ptr<tflite::Interpreter>> interpreter =
      interpreter_handler_->PrepareInterpreter(
          tflite::tools::CreateNullDelegate());
  if (interpreter == std::nullopt) {
    TFLITE_LOG(ERROR) << "Failed to prepare interpreter";
    return kTfLiteError;
  }
  interpreter_ = std::move(*interpreter);
  TFLITE_LOG(INFO) << "Reference interpreter prepared";
  std::optional<std::unique_ptr<ModelMetadata>> model_metadata =
      tflite::tooling::ModelMetadata::Create(interpreter_.get());
  if (model_metadata == std::nullopt) {
    TFLITE_LOG(ERROR) << "Failed to create model info";
    return kTfLiteError;
  }
  model_metadata_ = std::move(*model_metadata);

  input_manager_ = std::make_unique<TfliteInputManager>(interpreter_.get());
  if (input_manager_->PrepareInputData() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare input data";
    return kTfLiteError;
  }
  if (interpreter_handler_->RunInference(interpreter_.get(),
                                         input_manager_.get()) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to run reference inference";
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Reference inference run completed!";
  return kTfLiteOk;
}

bool CulpritFinder::CulpritSearchMatchCondition(OverallStat overall_stat) {
  bool condition = false;
  if (params_.Get<bool>(kFindNanFlag) &&
      !overall_stat.nan_output_indices.empty()) {
    condition |= true;
  }
  if (params_.Get<bool>(kFindNumericErrorFlag) &&
      overall_stat.total_error >= params_.Get<float>(kMinNumericErrorFlag)) {
    condition |= true;
  }
  if (overall_stat.is_crash) {
    condition |= true;
  }
  return condition;
}

TfLiteStatus CulpritFinder::RunCulpritFinderBinarySearch() {
  if (PrepareCulpritFinder() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare culprit finder";
    return kTfLiteError;
  }
  int start_node = 0;
  int end_node = interpreter_->nodes_size();

  OverallStat temp_overall_stat;
  if (CalculateErrorStats(start_node, end_node, &temp_overall_stat) !=
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
    int mid_node =
        std::floor((start_node_range_start + start_node_range_end) / 2);
    TFLITE_LOG(INFO) << "Looking for start node in node range: ["
                     << start_node_range_start << " - " << start_node_range_end
                     << "] by computing error stats for range [" << mid_node
                     << " - " << end_node << "]";
    auto output_tensors = model_metadata_->GetOutputTensorsOfNode(end_node);
    if (CalculateErrorStats(mid_node, end_node, &overall_stat,
                            output_tensors) != kTfLiteOk) {
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
    int mid_node = std::floor((end_node_range_start + end_node_range_end) / 2);
    TFLITE_LOG(INFO) << "Looking for end node in node range: ["
                     << end_node_range_start << " - " << end_node_range_end
                     << "] by computing error stats for range [" << start_node
                     << " - " << mid_node << "]";
    auto output_tensors = model_metadata_->GetOutputTensorsOfNode(mid_node);
    if (CalculateErrorStats(start_node, mid_node, &overall_stat,
                            output_tensors) != kTfLiteOk) {
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

TfLiteStatus CulpritFinder::RunCulpritFinderLinearSearch() {
  if (PrepareCulpritFinder() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare culprit finder";
    return kTfLiteError;
  }
  std::vector<std::pair<int, int>> node_ranges;
  node_ranges.reserve(interpreter_->nodes_size());
  TFLITE_LOG(INFO) << "Nodes size: " << interpreter_->nodes_size();
  TFLITE_LOG(INFO) << "Subgraphs size: " << interpreter_->subgraphs_size();
  auto execution_plan = interpreter_->execution_plan();
  TFLITE_LOG(INFO) << "Execution plan size: " << execution_plan.size();
  int batch_size = params_.Get<int>(kLinearSearchBatchSizeFlag);
  std::unordered_set<std::string> filter_node_names(
      absl::StrSplit(params_.Get<std::string>(kLinearSearchNodeFilterFlag), ',',
                     absl::SkipEmpty()));
  for (int i = 0; i < execution_plan.size() - batch_size + 1; ++i) {
    if (!filter_node_names.empty() &&
        filter_node_names.find(model_metadata_->GetNodeIdentifier(
            execution_plan[i])) == filter_node_names.end()) {
      continue;
    }
    node_ranges.push_back(
        {execution_plan[i], execution_plan[i + batch_size - 1]});
  }
  TFLITE_LOG(INFO) << "### Node ranges size: " << node_ranges.size();
  for (const auto& node_range : node_ranges) {
    OverallStat overall_stat;
    auto output_tensors =
        model_metadata_->GetOutputTensorsOfNode(node_range.second);
    if (CalculateErrorStats(node_range.first, node_range.second, &overall_stat,
                            output_tensors) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to calculate error stats";
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      overall_stats_.push_back({overall_stat.total_error, overall_stat});
    }
    TFLITE_LOG(INFO) << "Done with Node range: [" << node_range.first << " - "
                     << node_range.second << "]";
  }
  MakeReport();
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::NodeRangeAnalysis(int start_node, int end_node) {
  // Once we find the smallest node range that causes a NaN/NumericDifference,
  // we want to drill down into each node in the range and see if we can narrow
  // down the culprit further. We can do this by looking at output tensors of
  // each individual node in the range. To do the above we simply add the output
  // tensors of the specific node as the output tensor of the model. This does 2
  // things.
  // 1. It makes the output tensor of the model available as the output of the
  // model for inspection.
  // 2. It also splits the delegate into 2 parts. One that contains the nodes
  // before the node of interest and another of the nodes after the node of
  // interest.
  // We then run the model with the resulting delegate(s) and see if the NaN
  // still persists. If it does, it means that this is a valid split and this
  // specific node is not fused on the delegate side. If it doesn't, it means
  // we can ignore the output tensors generated for this node and continue to
  // the next node. This process can be repeated multiple times to see which
  // node generates how much deviation.

  TFLITE_LOG(INFO) << "Beginning NodeRangeAnalysis: " << start_node << " - "
                   << end_node;
  auto node_ids = model_metadata_->GetNodeIdsInRange(start_node, end_node);
  auto all_output_tensors = std::vector<int>();
  for (int node_id : node_ids) {
    auto output_tensors = model_metadata_->GetOutputTensorsOfNode(node_id);
    all_output_tensors.insert(all_output_tensors.end(), output_tensors.begin(),
                              output_tensors.end());
  }

  for (int node_id : node_ids) {
    auto output_tensors = model_metadata_->GetOutputTensorsOfNode(node_id);
    OverallStat overall_stat;
    if (CalculateErrorStats(start_node, node_id, &overall_stat,
                            output_tensors) != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to calculate error stats";
      return kTfLiteError;
    }

    if (CulpritSearchMatchCondition(overall_stat)) {
      LogOverallStat(overall_stat);
    }

    TFLITE_LOG(INFO) << "Done with Node range: [" << start_node << " - "
                     << node_id << "] with node: " << node_id;
  }

  return kTfLiteOk;
}

inline void CulpritFinder::LogOverallStat(const OverallStat& overall_stat) {
  TFLITE_LOG(INFO) << "Overall stat:";

  if (overall_stat.delegated_node_range.first !=
      overall_stat.delegated_node_range.second) {
    TFLITE_LOG(INFO) << "  Delegated node range: ["
                     << model_metadata_->GetNodeIdentifier(
                            overall_stat.delegated_node_range.first, true)
                     << ", "
                     << model_metadata_->GetNodeIdentifier(
                            overall_stat.delegated_node_range.second, true)
                     << "]";
  } else {
    TFLITE_LOG(INFO) << "  Delegated node range: "
                     << model_metadata_->GetNodeIdentifier(
                            overall_stat.delegated_node_range.first, true);
  }
  TFLITE_LOG(INFO) << "  Min elementwise error: " << overall_stat.min_error;
  TFLITE_LOG(INFO) << "  Max elementwise error: " << overall_stat.max_error;
  TFLITE_LOG(INFO) << "  Total average error: " << overall_stat.total_error;
  TFLITE_LOG(INFO) << "  NAN output indices: ";
  for (int nan_output_index : overall_stat.nan_output_indices) {
    TFLITE_LOG(INFO) << model_metadata_->GetTensorIdentifier(nan_output_index)
                     << ", ";
  }
}

void CulpritFinder::MakeReport() {
  std::sort(
      overall_stats_.begin(), overall_stats_.end(),
      [](const std::pair<float, OverallStat>& a,
         const std::pair<float, OverallStat>& b) { return a.first > b.first; });

  std::unordered_map<std::string, std::vector<int>> node_to_overall_stats_index;
  std::unordered_map<std::string, std::vector<int>>
      node_to_overall_stats_index_with_nan;
  for (int i = 0; i < overall_stats_.size(); ++i) {
    node_to_overall_stats_index[model_metadata_->GetNodeIdentifier(
                                    overall_stats_[i]
                                        .second.delegated_node_range.first)]
        .push_back(i);
    if (!overall_stats_[i].second.nan_output_indices.empty()) {
      node_to_overall_stats_index_with_nan
          [model_metadata_->GetNodeIdentifier(
               overall_stats_[i].second.delegated_node_range.first)]
              .push_back(i);
    }
  }

  std::vector<std::pair<std::string, std::vector<int>>>
      sorted_node_to_overall_stats_index(node_to_overall_stats_index.begin(),
                                         node_to_overall_stats_index.end());
  std::sort(sorted_node_to_overall_stats_index.begin(),
            sorted_node_to_overall_stats_index.end(),
            [](const std::pair<std::string, std::vector<int>>& a,
               const std::pair<std::string, std::vector<int>>& b) {
              return a.second.size() > b.second.size();
            });
  std::vector<std::pair<std::string, std::vector<int>>>
      sorted_node_to_overall_stats_index_with_nan(
          node_to_overall_stats_index_with_nan.begin(),
          node_to_overall_stats_index_with_nan.end());
  std::sort(sorted_node_to_overall_stats_index_with_nan.begin(),
            sorted_node_to_overall_stats_index_with_nan.end(),
            [](const std::pair<std::string, std::vector<int>>& a,
               const std::pair<std::string, std::vector<int>>& b) {
              return a.second.size() > b.second.size();
            });

  TFLITE_LOG(INFO) << "CULPRIT FINDER REPORT";
  TFLITE_LOG(INFO)
      << "-------------------------------------------------------------";
  TFLITE_LOG(INFO) << "Total number of nodes with errors: "
                   << overall_stats_.size();
  TFLITE_LOG(INFO)
      << "Top 5 node ranges sorted by error (node_range, op_name(s), "
         "input/output shapes, total_error):";
  for (int i = 0; i < overall_stats_.size() && i < 5; ++i) {
    int node_start_index = overall_stats_[i].second.delegated_node_range.first;
    int node_end_index = overall_stats_[i].second.delegated_node_range.second;

    TFLITE_LOG(INFO) << node_start_index << " - " << node_end_index << ", "
                     << model_metadata_->GetNodeIdentifier(node_start_index)
                     << " - "
                     << model_metadata_->GetNodeIdentifier(node_end_index)
                     << ", " << model_metadata_->GetNodeShapes(node_start_index)
                     << ", " << overall_stats_[i].first;
  }

  TFLITE_LOG(INFO)
      << "-------------------------------------------------------------";
  TFLITE_LOG(INFO) << "Top 5 node(s) with most errors (op_name, count):";
  for (int i = 0; i < sorted_node_to_overall_stats_index.size() && i < 5; ++i) {
    TFLITE_LOG(INFO) << sorted_node_to_overall_stats_index[i].first << ", "
                     << sorted_node_to_overall_stats_index[i].second.size();
  }

  TFLITE_LOG(INFO)
      << "-------------------------------------------------------------";
  if (!sorted_node_to_overall_stats_index_with_nan.empty()) {
    TFLITE_LOG(INFO)
        << "Top 5 Node(s) signatures with most nans (op_name, count):";
    for (int i = 0;
         i < sorted_node_to_overall_stats_index_with_nan.size() && i < 5; ++i) {
      TFLITE_LOG(INFO)
          << sorted_node_to_overall_stats_index_with_nan[i].first << ", "
          << sorted_node_to_overall_stats_index_with_nan[i].second.size();
    }
  }
}

}  // namespace tooling
}  // namespace tflite
