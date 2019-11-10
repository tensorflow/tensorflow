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

#include <sstream>

#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace profiling {
namespace {

struct OperatorDetails {
  uint32_t subgraph_index;
  uint32_t node_index;
  std::string name;
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
  int code = node_reg->second.builtin_code;
  const char* op_name = nullptr;
  if (code == tflite::BuiltinOperator_CUSTOM) {
    const char* custom_name = node_reg->second.custom_name;
    op_name = custom_name ? custom_name : "UnknownCustomOp";
  } else {
    op_name = tflite::EnumNamesBuiltinOperator()[code];
  }
  const char* profiling_string =
      interpreter.OpProfilingString(node_reg->second, &node_reg->first);
  OperatorDetails details;
  details.name = op_name;
  if (profiling_string) {
    details.name += ":" + std::string(profiling_string);
  }
  details.inputs = GetTensorNames(interpreter, inputs);
  details.outputs = GetTensorNames(interpreter, outputs);
  return details;
}

tensorflow::StatSummarizerOptions GetProfileSummarizerOptions() {
  auto options = tensorflow::StatSummarizerOptions();
  // Summary will be manually handled per subgraphs in order to keep the
  // compatibility.
  options.show_summary = false;
  options.show_memory = false;
  return options;
}

}  // namespace

ProfileSummarizer::ProfileSummarizer() {
  // Create stats calculator for the primary graph.
  stats_calculator_map_[0] = std::unique_ptr<tensorflow::StatsCalculator>(
      new tensorflow::StatsCalculator(GetProfileSummarizerOptions()));
}

void ProfileSummarizer::ProcessProfiles(
    const std::vector<const ProfileEvent*>& profile_stats,
    const tflite::Interpreter& interpreter) {
  if (profile_stats.empty()) return;

  std::vector<const ProfileEvent*> events;
  std::copy_if(profile_stats.begin(), profile_stats.end(),
               std::back_inserter(events), [](const ProfileEvent* e) {
                 return e->end_timestamp_us >= e->begin_timestamp_us;
               });
  // Sort with begin_time.
  std::sort(events.begin(), events.end(),
            [](const ProfileEvent* const& a, const ProfileEvent* const& b) {
              return a->begin_timestamp_us < b->begin_timestamp_us;
            });
  if (events.empty()) {
    return;
  }

  int64_t base_start_us = events[0]->begin_timestamp_us;
  int node_num = 0;
  auto tag_string = [](const string& s, const string& t) {
    return (t == "OpInvoke" || t == "DelegateOpInvoke") ? s : s + "/" + t;
  };

  // Total time will be accumulated per subgraph.
  std::map<uint32_t, int64_t> total_us_per_subgraph_map;

  for (auto event : events) {
    const auto subgraph_index = event->event_subgraph_index;
    auto stats_calculator = GetStatsCalculator(subgraph_index);
    int64_t start_us = event->begin_timestamp_us - base_start_us;
    int64_t node_exec_time =
        event->end_timestamp_us - event->begin_timestamp_us;
    if (event->event_type == Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      // When recording an OPERATOR_INVOKE_EVENT, we have recorded the node
      // index as event_metadata. See the macro
      // TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE defined in
      // tensorflow/lite/core/api/profiler.h for details.
      const auto node_index = event->event_metadata;

      const auto op_details =
          GetOperatorDetails(interpreter, subgraph_index, node_index);
      const auto type_in_stats = tag_string(op_details.name, event->tag);

      const auto node_name = ToString(op_details.outputs);
      // Append node index to node name because 'stats_calculator' can not
      // distinguish two nodes w/ the same 'node_name'.
      const auto node_name_in_stats =
          tag_string(node_name + ":" + std::to_string(node_index), event->tag);

      stats_calculator->AddNodeStats(node_name_in_stats, type_in_stats,
                                     node_num, start_us, node_exec_time,
                                     0 /*memory */);
    } else {
      // TODO(b/139812778) consider use a different stats_calculator to record
      // non-op-invoke events so that these could be separated from
      // op-invoke-events in the final profiling stats report.
      const memory::MemoryUsage node_mem_usage =
          event->end_mem_usage - event->begin_mem_usage;
      std::string node_name(event->tag);
      node_name += "/" + std::to_string(event->event_subgraph_index);
      stats_calculator->AddNodeStats(node_name, "Misc Runtime Ops", node_num,
                                     start_us, node_exec_time,
                                     node_mem_usage.total_allocated_bytes);
    }

    // Add total time except actual delegate ops since the elapsed time of the
    // delegate ops inside are already combined at a fused DELEGATE op.
    if (strcmp(event->tag, "DelegateOpInvoke") != 0) {
      total_us_per_subgraph_map[subgraph_index] += node_exec_time;
    }
    ++node_num;
  }

  for (auto& total_us_per_subgraph_pair : total_us_per_subgraph_map) {
    auto stats_calculator =
        GetStatsCalculator(total_us_per_subgraph_pair.first);
    stats_calculator->UpdateRunTotalUs(total_us_per_subgraph_pair.second);
  }
}

tensorflow::StatsCalculator* ProfileSummarizer::GetStatsCalculator(
    uint32_t subgraph_index) {
  if (stats_calculator_map_.count(subgraph_index) == 0) {
    stats_calculator_map_[subgraph_index] =
        std::unique_ptr<tensorflow::StatsCalculator>(
            new tensorflow::StatsCalculator(GetProfileSummarizerOptions()));
  }
  return stats_calculator_map_[subgraph_index].get();
}

std::string ProfileSummarizer::GenerateReport(std::string tag,
                                              bool include_output_string) {
  std::stringstream stream;
  bool has_non_primary_graph =
      (stats_calculator_map_.size() - stats_calculator_map_.count(0)) > 0;
  for (auto& stats_calc : stats_calculator_map_) {
    auto subgraph_index = stats_calc.first;
    auto subgraph_stats = stats_calc.second.get();
    if (has_non_primary_graph) {
      if (subgraph_index == 0)
        stream << "Primary graph " << tag << ":" << std::endl;
      else
        stream << "Subgraph (index: " << subgraph_index << ") " << tag << ":"
               << std::endl;
    }
    if (include_output_string) {
      stream << subgraph_stats->GetOutputString();
    }
    if (subgraph_index != 0) {
      stream << "Subgraph (index: " << subgraph_index << ") ";
    }
    stream << subgraph_stats->GetShortSummary() << std::endl;
  }
  return stream.str();
}

}  // namespace profiling
}  // namespace tflite
