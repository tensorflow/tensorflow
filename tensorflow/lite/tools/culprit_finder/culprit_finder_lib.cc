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
#include <cfloat>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/profiling/model_runtime_info.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/culprit_finder/culprit_finder_utils.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/model_loader.h"
#include "tensorflow/lite/tools/tool_params.h"
#include "tensorflow/lite/tools/utils.h"

namespace tflite {
namespace tooling {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kSearchStrategyFlag[] = "search_strategy";
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
// Input data range specific flags.
constexpr char kInputDataRangeMinFlag[] = "input_data_range_min";
constexpr char kInputDataRangeMaxFlag[] = "input_data_range_max";

// Search strategy enums.
constexpr char kBinarySearchStrategyEnum[] = "binary";
constexpr char kLinearSearchStrategyEnum[] = "linear";

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
  ABSL_CHECK_OK(LoadModel());
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
  params_.AddParam(kBinarySearchReverseSweepFlag,
                   ToolParam::Create<bool>(false));
  params_.AddParam(kLinearSearchBatchSizeFlag, ToolParam::Create<int>(1));
  params_.AddParam(kLinearSearchNodeFilterFlag,
                   ToolParam::Create<std::string>(""));
  params_.AddParam(kFindNanFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kFindNumericErrorFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kMinNumericErrorFlag, ToolParam::Create<float>(0.0001));
  params_.AddParam(kGPUPrecisionLossAllowedFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kInputDataRangeMinFlag, ToolParam::Create<float>(FLT_MIN));
  params_.AddParam(kInputDataRangeMaxFlag, ToolParam::Create<float>(FLT_MAX));
  delegate_list_util_.AddAllDelegateParams();
}

void CulpritFinder::LogParams() {
  LOG_TOOL_PARAM(params_, std::string, kModelFileFlag, "Model file", true);
  LOG_TOOL_PARAM(params_, std::string, kSearchStrategyFlag, "Search strategy",
                 true);
  LOG_TOOL_PARAM(params_, bool, kBinarySearchReverseSweepFlag,
                 "Binary search find end first", true);
  LOG_TOOL_PARAM(params_, int, kLinearSearchBatchSizeFlag,
                 "Linear search batch size", true);
  LOG_TOOL_PARAM(params_, std::string, kLinearSearchNodeFilterFlag,
                 "Linear search node filter", true);
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
      CreateFlag<bool>(kBinarySearchReverseSweepFlag, &params_,
                       "If true, find the end node first. Default is false."),
      CreateFlag<int>(kLinearSearchBatchSizeFlag, &params_,
                      "If provided, the culprit finder will run the linear "
                      "search for batches of this size."),
      CreateFlag<std::string>(
          kLinearSearchNodeFilterFlag, &params_,
          "A comma-separated list of node types to filter out."),
      CreateFlag<bool>(kFindNanFlag, &params_,
                       "If specified, searches for NANs."),
      CreateFlag<bool>(kFindNumericErrorFlag, &params_,
                       "If specified, searches for numeric errors."),
      CreateFlag<float>(kMinNumericErrorFlag, &params_,
                        "Minimum absolute difference to consider an "
                        "inference as an error."),
      CreateFlag<bool>(kGPUPrecisionLossAllowedFlag, &params_,
                       "Allow GPU precision loss."),
      CreateFlag<float>(kInputDataRangeMinFlag, &params_,
                        "Minimum value of the input data range."),
      CreateFlag<float>(kInputDataRangeMaxFlag, &params_,
                        "Maximum value of the input data range."),
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

absl::Status CulpritFinder::LoadModel() {
  model_loader_ =
      std::make_unique<::tflite::tools::PathModelLoader>(GetModelPath());
  if (!model_loader_) {
    TFLITE_LOG(ERROR) << "Failed to initialize model loader with path "
                      << GetModelPath();
    return absl::InternalError("Failed to initialize model loader");
  }
  if (!model_loader_->Init()) {
    TFLITE_LOG(ERROR) << "Failed to load model " << GetModelPath();
    return absl::InternalError("Failed to load model");
  }
  model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(
          model_loader_->GetModel()->allocation()->base()),
      model_loader_->GetModel()->allocation()->bytes());
  TFLITE_LOG(INFO) << "Loaded model: " << GetModelPath();
  return absl::OkStatus();
}

TfLiteStatus CulpritFinder::PrepareInputData() {
  if (interpreter_ == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return kTfLiteError;
  }
  inputs_data_.clear();
  for (int input_index : interpreter_->inputs()) {
    TfLiteTensor* input_tensor = interpreter_->tensor(input_index);
    if (input_tensor->type == kTfLiteString) {
      TFLITE_LOG(ERROR) << "String input tensor is not supported";
      return kTfLiteError;
    }
    float low_range = 0;
    float high_range = 0;
    tflite::utils::GetDataRangesForType(input_tensor->type, &low_range,
                                        &high_range);
    if (params_.HasValueSet<float>(kInputDataRangeMinFlag)) {
      low_range = params_.Get<float>(kInputDataRangeMinFlag);
    }
    if (params_.HasValueSet<float>(kInputDataRangeMaxFlag)) {
      high_range = params_.Get<float>(kInputDataRangeMaxFlag);
    }

    tflite::utils::InputTensorData input_data =
        tflite::utils::CreateRandomTensorData(*input_tensor, low_range,
                                              high_range);
    inputs_data_.push_back(std::move(input_data));
  }
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::SetInputTensors(tflite::Interpreter* interpreter) {
  if (interpreter == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return kTfLiteError;
  }

  for (int i = 0; i < interpreter->inputs().size(); ++i) {
    int input_index = interpreter->inputs()[i];
    TfLiteTensor* input_tensor = interpreter->tensor(input_index);
    if (input_tensor->type == kTfLiteString) {
      TFLITE_LOG(ERROR) << "String input tensor is not supported";
      return kTfLiteError;
    }
    std::memcpy(input_tensor->data.raw, inputs_data_[i].data.get(),
                inputs_data_[i].bytes);
  }
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::PrepareInterpreter(
    tflite::tools::TfLiteDelegatePtr delegate,
    std::vector<int> intermediate_outputs) {
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<tflite::Interpreter>* interpreter = nullptr;
  tflite::InterpreterOptions options;
  if (delegate.get() != nullptr) {
    interpreter = &interpreter_with_delegate_;
  } else {
    options.SetPreserveAllTensors(true);
    interpreter = &interpreter_;
  }

  tflite::InterpreterBuilder interpreter_builder(*model_, resolver, &options);
  if (interpreter_builder(interpreter) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to build interpreter";
    return kTfLiteError;
  }
  if (*interpreter == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return kTfLiteError;
  }

  if (!intermediate_outputs.empty()) {
    auto outputs = (*interpreter)->outputs();
    outputs.insert(outputs.end(), intermediate_outputs.begin(),
                   intermediate_outputs.end());
    (*interpreter)->SetOutputs(outputs);
  }

  if (delegate.get() != nullptr) {
    (*interpreter)->ModifyGraphWithDelegate(delegate.get());
    owned_delegates_.push_back(std::move(delegate));
  }

  if ((*interpreter)->AllocateTensors() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to allocate tensors";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::RunInference(tflite::Interpreter* interpreter) {
  if (interpreter == nullptr) {
    TFLITE_LOG(ERROR) << "Interpreter is null";
    return kTfLiteError;
  }
  interpreter->ResetVariableTensors();
  if (SetInputTensors(interpreter) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to set input tensors";
    return kTfLiteError;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to invoke interpreter";
    return kTfLiteError;
  }
  return kTfLiteOk;
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
    auto status = PrepareInterpreter(GetDelegate(start_node, end_node),
                                     intermediate_outputs);
    if (status != kTfLiteOk) {
      TFLITE_LOG(ERROR) << "Failed to prepare interpreter";
      return kTfLiteError;
    }

    status = RunInference(interpreter_with_delegate_.get());
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

TfLiteStatus CulpritFinder::PopulateModelRuntimeDetails() {
  tflite::profiling::ModelRuntimeDetails model_runtime_details;
  if (tflite::profiling::GenerateModelRuntimeInfo(
          *interpreter_, model_runtime_details) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to generate model runtime info";
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Model runtime info generated";

  for (const auto& subgraph : model_runtime_details.subgraphs()) {
    if (subgraph.subgraph_id() == 0) {
      for (const auto& edge : subgraph.edges()) {
        tensor_index_to_edge_proto_[edge.id()] = edge;
      }
      for (const auto& node : subgraph.nodes()) {
        node_index_to_node_proto_[node.id()] = node;
        for (int input : node.inputs()) {
          tensor_index_to_dst_nodes_[input].push_back(node.id());
        }
        for (int output : node.outputs()) {
          tensor_index_to_src_nodes_[output] = node.id();
        }
      }
      break;
    }
  }

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
    } else {
      MakeReport();
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
  if (PrepareInterpreter(tflite::tools::CreateNullDelegate()) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare interpreter";
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Reference interpreter prepared";
  if (PopulateModelRuntimeDetails() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to populate model runtime details";
    return kTfLiteError;
  }
  if (PrepareInputData() != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Failed to prepare input data";
    return kTfLiteError;
  }
  if (RunInference(interpreter_.get()) != kTfLiteOk) {
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
        filter_node_names.find(GetNodeIdentifier(execution_plan[i])) ==
            filter_node_names.end()) {
      continue;
    }
    node_ranges.push_back(
        {execution_plan[i], execution_plan[i + batch_size - 1]});
  }
  TFLITE_LOG(INFO) << "### Node ranges size: " << node_ranges.size();
  for (const auto& node_range : node_ranges) {
    OverallStat overall_stat;
    auto output_tensors = GetOutputTensorsOfNode(node_range.second);
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
  return kTfLiteOk;
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

  NodeRangeAnalysis(start_node, end_node);

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
    auto output_tensors = GetOutputTensorsOfNode(end_node);
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
    auto output_tensors = GetOutputTensorsOfNode(mid_node);
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
  auto node_ids = GetNodeIdsInRange(start_node, end_node);
  auto all_output_tensors = std::vector<int>();
  for (int node_id : node_ids) {
    auto output_tensors = GetOutputTensorsOfNode(node_id);
    all_output_tensors.insert(all_output_tensors.end(), output_tensors.begin(),
                              output_tensors.end());
  }

  for (int node_id : node_ids) {
    auto output_tensors = GetOutputTensorsOfNode(node_id);
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
                     << GetNodeIdentifier(
                            overall_stat.delegated_node_range.first, true)
                     << ", "
                     << GetNodeIdentifier(
                            overall_stat.delegated_node_range.second, true)
                     << "]";
  } else {
    TFLITE_LOG(INFO) << "  Delegated node range: "
                     << GetNodeIdentifier(
                            overall_stat.delegated_node_range.first, true);
  }
  TFLITE_LOG(INFO) << "  Min elementwise error: " << overall_stat.min_error;
  TFLITE_LOG(INFO) << "  Max elementwise error: " << overall_stat.max_error;
  TFLITE_LOG(INFO) << "  Total average error: " << overall_stat.total_error;
  TFLITE_LOG(INFO) << "  NAN output indices: ";
  for (int nan_output_index : overall_stat.nan_output_indices) {
    TFLITE_LOG(INFO) << GetTensorIdentifier(nan_output_index) << ", ";
  }
}

std::vector<int> CulpritFinder::GetNodeIdsInRange(int start_node,
                                                  int end_node) {
  std::vector<int> node_ids;
  for (int node_id : interpreter_->execution_plan()) {
    if (node_id >= start_node && node_id <= end_node) {
      node_ids.push_back(node_id);
    }
  }
  return node_ids;
}

std::vector<int> CulpritFinder::GetOutputTensorsOfNode(int node_id) {
  std::vector<int> output_tensors;
  for (int output_tensor_id : node_index_to_node_proto_[node_id].outputs()) {
    output_tensors.push_back(output_tensor_id);
  }
  return output_tensors;
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
    node_to_overall_stats_index
        [GetNodeIdentifier(overall_stats_[i].second.delegated_node_range.first)]
            .push_back(i);
    if (!overall_stats_[i].second.nan_output_indices.empty()) {
      node_to_overall_stats_index_with_nan
          [GetNodeIdentifier(
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
                     << GetNodeIdentifier(node_start_index) << " - "
                     << GetNodeIdentifier(node_end_index) << ", "
                     << GetNodeShapes(node_start_index) << ", "
                     << overall_stats_[i].first;
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

std::string CulpritFinder::EdgeShapeToString(
    const tflite::profiling::Edge& edge) {
  std::string shape_string = "";
  for (const auto& shape : edge.shape()) {
    shape_string += std::to_string(shape) + ",";
  }
  return tflite::profiling::Edge::DataType_Name(edge.data_type()) + "[" +
         shape_string + "]";
}

std::string CulpritFinder::GetNodeIdentifier(int node_index, bool with_index) {
  auto node_proto = node_index_to_node_proto_[node_index];
  if (with_index) {
    return absl::StrFormat("[%s]:%d", node_proto.name(), node_index);
  }
  return node_proto.name();
}

std::string CulpritFinder::GetTensorIdentifier(int tensor_index) {
  auto tensor_proto = tensor_index_to_edge_proto_[tensor_index];
  if (tensor_index_to_src_nodes_.find(tensor_index) ==
      tensor_index_to_src_nodes_.end()) {
    // This is an input tensor.
    return absl::StrFormat("(INPUT)->%d", tensor_index);
  }
  return absl::StrFormat(
      "(%s)->%d",
      GetNodeIdentifier(tensor_index_to_src_nodes_[tensor_index],
                        /*with_index=*/true),
      tensor_index);
}

std::string CulpritFinder::GetNodeShapes(int node_index) {
  auto node_proto = node_index_to_node_proto_[node_index];
  std::string input_shapes = "";
  for (const auto& input : node_proto.inputs()) {
    auto input_edge = tensor_index_to_edge_proto_[input];
    if (input_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
      continue;
    }
    input_shapes += EdgeShapeToString(input_edge);
    if (input != node_proto.inputs().size() - 1) input_shapes += ",";
  }
  std::string output_shapes = "";
  for (const auto& output : node_proto.outputs()) {
    auto output_edge = tensor_index_to_edge_proto_[output];
    if (output_edge.data_type() == tflite::profiling::Edge::UNKNOWN_TYPE) {
      continue;
    }
    output_shapes += EdgeShapeToString(output_edge);
    if (output != node_proto.outputs().size() - 1) output_shapes += ",";
  }
  return "(" + input_shapes + ") -> (" + output_shapes + ")";
}
}  // namespace tooling
}  // namespace tflite
