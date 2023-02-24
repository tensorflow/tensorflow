/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profile_summarizer.h"

#include <memory>
#include <sstream>
#include <string>

#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace profiling {
namespace {

struct OperatorDetails {
  uint32_t subgraph_index;
  uint32_t node_index;
  std::string op_description;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

std::string GetTensorName(const tflite::Interpreter& interpreter,
                          int tensor_index) {
  const auto tensor = interpreter.tensor(tensor_index);
  if (tensor == nullptr || tensor->name == nullptr) {
    return "Unknown";
  }
  return tensor->name;
}
std::vector<std::string> GetTensorNames(const tflite::Interpreter& interpreter,
                                        const TfLiteIntArray* tensor_indices) {
  std::vector<std::string> tensors;
  tensors.reserve(tensor_indices->size);
  for (int i = 0; i < tensor_indices->size; i++) {
    tensors.push_back(GetTensorName(interpreter, tensor_indices->data[i]));
  }
  return tensors;
}

std::string ToString(const std::vector<std::string>& str_vector) {
  std::stringstream stream;
  stream << "[";
  bool first = true;
  for (const auto& s : str_vector) {
    if (!first) {
      stream << ", ";
    } else {
      first = false;
    }
    stream << s;
  }
  stream << "]";
  return stream.str();
}

OperatorDetails GetOperatorDetails(const tflite::Interpreter& interpreter,
                                   uint32_t subgraph_index,
                                   uint32_t node_index) {
  auto subgraph =
      const_cast<tflite::Interpreter&>(interpreter).subgraph(subgraph_index);
  auto node_reg = subgraph->node_and_registration(node_index);
  auto inputs = node_reg->first.inputs;
  auto outputs = node_reg->first.outputs;
  const char* profiling_string =
      interpreter.OpProfilingString(node_reg->second, &node_reg->first);
  OperatorDetails details;
  if (profiling_string) {
    details.op_description = std::string(profiling_string);
  }
  details.inputs = GetTensorNames(interpreter, inputs);
  details.outputs = GetTensorNames(interpreter, outputs);
  return details;
}

}  // namespace

ProfileSummarizer::ProfileSummarizer(
    std::shared_ptr<ProfileSummaryFormatter> summary_formatter)
    : summary_formatter_(summary_formatter) {
  // Create stats calculator for the primary graph.
  stats_calculator_map_[0] = std::make_unique<tensorflow::StatsCalculator>(

      summary_formatter_->GetStatSummarizerOptions());

  // Create stats calculator for the delegation op.
  delegate_stats_calculator_ = std::make_unique<tensorflow::StatsCalculator>(

      summary_formatter_->GetStatSummarizerOptions());
}
void ProfileSummarizer::ProcessProfiles(
    const std::vector<const ProfileEvent*>& profile_stats,
    const tflite::Interpreter& interpreter) {
  if (profile_stats.empty()) return;

  int node_num = 0;

  // Total time will be accumulated per subgraph.
  std::map<uint32_t, int64_t> total_us_per_subgraph_map;
  int64_t delegate_internal_total_us = 0;

  for (auto event : profile_stats) {
    const auto subgraph_index = event->extra_event_metadata;
    auto stats_calculator = GetStatsCalculator(subgraph_index);
    int64_t node_exec_time = event->elapsed_time;
    if (event->event_type == Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      // When recording an OPERATOR_INVOKE_EVENT, we have recorded the node
      // index as event_metadata. See the macro
      // TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE defined in
      // tensorflow/lite/core/api/profiler.h for details.
      const auto node_index = event->event_metadata;

      const auto op_details =
          GetOperatorDetails(interpreter, subgraph_index, node_index);
      std::string type_in_stats(event->tag);
      if (!op_details.op_description.empty()) {
        type_in_stats += "/" + op_details.op_description;
      }

      const auto node_name = ToString(op_details.outputs);
      // Append node index to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          node_name + ":" + std::to_string(node_index);

      stats_calculator->AddNodeStats(node_name_in_stats, type_in_stats,
                                     node_num, node_exec_time, 0 /*memory */);
    } else if (event->event_type ==
               Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      const std::string node_name(event->tag);
      // Append event_metadata to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          "Delegate/" + node_name + ":" + std::to_string(event->event_metadata);

      delegate_stats_calculator_->AddNodeStats(node_name_in_stats,
                                               "DelegateOpInvoke", node_num,
                                               node_exec_time, 0 /*memory */);
    } else if (event->event_type ==
               Profiler::EventType::DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT) {
      // This event type handles the delegate ops that are profiled in the
      // Operator-wise Profiling section, not in the Delegate internal section.
      const std::string node_name(event->tag);
      // For delegate op, node name is treated as the type in stats.
      const std::string type_in_stats(node_name);
      // Append event_metadata to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          "Delegate/" + node_name + ":" + std::to_string(event->event_metadata);

      stats_calculator->AddNodeStats(node_name_in_stats, type_in_stats,
                                     node_num, node_exec_time, 0 /*memory */);
    } else {
      // Note: a different stats_calculator could be used to record
      // non-op-invoke events so that these could be separated from
      // op-invoke-events in the final profiling stats report.
      const memory::MemoryUsage node_mem_usage =
          event->end_mem_usage - event->begin_mem_usage;
      std::string node_name(event->tag);
      if (node_name == "Invoke") {
        // Don't count the overall Invoke for profiling.
        continue;
      }
      node_name += "/" + std::to_string(event->extra_event_metadata);
      stats_calculator->AddNodeStats(node_name, event->tag, node_num,
                                     node_exec_time,
                                     node_mem_usage.mem_footprint_kb * 1000.0);
    }

    // Add total time except delegate ops that are profiled separately since the
    // elapsed time of the delegate ops inside are already combined at a fused
    // DELEGATE op.
    if (event->event_type !=
        Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      total_us_per_subgraph_map[subgraph_index] += node_exec_time;
    } else {
      delegate_internal_total_us += node_exec_time;
    }
    ++node_num;
  }

  for (auto& total_us_per_subgraph_pair : total_us_per_subgraph_map) {
    auto stats_calculator =
        GetStatsCalculator(total_us_per_subgraph_pair.first);
    stats_calculator->UpdateRunTotalUs(total_us_per_subgraph_pair.second);
  }
  if (delegate_internal_total_us > 0) {
    delegate_stats_calculator_->UpdateRunTotalUs(delegate_internal_total_us);
  }
}

tensorflow::StatsCalculator* ProfileSummarizer::GetStatsCalculator(
    uint32_t subgraph_index) {
  if (stats_calculator_map_.count(subgraph_index) == 0) {
    stats_calculator_map_[subgraph_index] =
        std::make_unique<tensorflow::StatsCalculator>(

            summary_formatter_->GetStatSummarizerOptions());
  }
  return stats_calculator_map_[subgraph_index].get();
}

}  // namespace profiling
}  // namespace tflite
