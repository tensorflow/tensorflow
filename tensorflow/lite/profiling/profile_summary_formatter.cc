/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/profiling/profile_summary_formatter.h"

#include <fstream>
#include <iomanip>
#include <ios>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/profiling/proto/profiling_info.pb.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace profiling {

std::string ProfileSummaryDefaultFormatter::GetOutputString(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator,
    const std::map<uint32_t, std::string>& subgraph_name_map) const {
  return GenerateReport("profile", /*include_output_string*/ true,
                        stats_calculator_map, delegate_stats_calculator,
                        subgraph_name_map);
}

std::string ProfileSummaryDefaultFormatter::GetShortSummary(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator,
    const std::map<uint32_t, std::string>& subgraph_name_map) const {
  return GenerateReport("summary", /*include_output_string*/ false,
                        stats_calculator_map, delegate_stats_calculator,
                        subgraph_name_map);
}

std::string ProfileSummaryDefaultFormatter::GenerateReport(
    const std::string& tag, bool include_output_string,
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator,
    const std::map<uint32_t, std::string>& subgraph_name_map) const {
  std::stringstream stream;
  bool has_non_primary_graph =
      (stats_calculator_map.size() - stats_calculator_map.count(0)) > 0;
  for (const auto& stats_calc : stats_calculator_map) {
    auto subgraph_index = stats_calc.first;
    auto subgraph_stats = stats_calc.second.get();
    std::string subgraph_name = "";
    if (subgraph_name_map.find(subgraph_index) != subgraph_name_map.end()) {
      subgraph_name = subgraph_name_map.at(subgraph_index);
    }

    if (has_non_primary_graph) {
      if (subgraph_index == 0) {
        stream << "Primary graph (name: " << subgraph_name << ") " << tag << ":"
               << std::endl;
      } else {
        stream << "Subgraph (index: " << subgraph_index
               << ", name: " << subgraph_name << ") " << tag << ":"
               << std::endl;
      }
    }
    if (include_output_string) {
      stream << subgraph_stats->GetOutputString();
    }
    if (subgraph_index != 0) {
      stream << "Subgraph (index: " << subgraph_index
             << ", name: " << subgraph_name << ") ";
    }
    stream << subgraph_stats->GetShortSummary() << std::endl;
  }

  if (delegate_stats_calculator.num_runs() > 0) {
    stream << "Delegate internal: " << std::endl;
    if (include_output_string) {
      stream << delegate_stats_calculator.GetOutputString();
    }
    stream << delegate_stats_calculator.GetShortSummary() << std::endl;
  }

  return stream.str();
}

void ProfileSummaryDefaultFormatter::HandleOutput(
    const std::string& init_output, const std::string& run_output,
    std::string output_file_path) const {
  std::ofstream output_file(output_file_path);
  std::ostream* output_stream = nullptr;
  if (output_file.good()) {
    output_stream = &output_file;
  }
  if (!init_output.empty()) {
    WriteOutput("Profiling Info for Benchmark Initialization:", init_output,
                output_stream == nullptr ? &TFLITE_LOG(INFO) : output_stream);
  }
  if (!run_output.empty()) {
    WriteOutput(
        "Operator-wise Profiling Info for Regular Benchmark Runs:", run_output,
        output_stream == nullptr ? &TFLITE_LOG(INFO) : output_stream);
  }
}

tensorflow::StatSummarizerOptions
ProfileSummaryDefaultFormatter::GetStatSummarizerOptions() const {
  auto options = tensorflow::StatSummarizerOptions();
  // Summary will be manually handled per subgraphs in order to keep the
  // compatibility.
  options.show_summary = false;
  options.show_memory = false;
  return options;
}

tensorflow::StatSummarizerOptions
ProfileSummaryCSVFormatter::GetStatSummarizerOptions() const {
  auto options = ProfileSummaryDefaultFormatter::GetStatSummarizerOptions();
  options.format_as_csv = true;
  return options;
}

std::vector<tensorflow::StatsCalculator::Detail>
ProfileSummaryProtoFormatter::GetDetailsSortedByRunOrder(
    const tensorflow::StatsCalculator* stats_calculator) const {
  std::vector<tensorflow::StatsCalculator::Detail> details;
  std::map<std::string, tensorflow::StatsCalculator::Detail> unsorted_details =
      stats_calculator->GetDetails();

  std::priority_queue<
      std::pair<std::string, const tensorflow::StatsCalculator::Detail*>>
      sorted_list;
  const int num_nodes = unsorted_details.size();
  for (const auto& det : unsorted_details) {
    const tensorflow::StatsCalculator::Detail* detail = &(det.second);
    std::stringstream stream_for_sort;
    stream_for_sort << std::setw(20) << std::right << std::setprecision(10)
                    << std::fixed;
    stream_for_sort << num_nodes - detail->run_order;
    sorted_list.emplace(stream_for_sort.str(), detail);
  }

  while (!sorted_list.empty()) {
    auto entry = sorted_list.top();
    sorted_list.pop();
    details.push_back(*entry.second);
  }
  return details;
}

void ProfileSummaryProtoFormatter::GenerateOpProfileDataFromDetail(
    const tensorflow::StatsCalculator::Detail* detail,
    const tensorflow::StatsCalculator* stats_calculator,
    OpProfileData* const op_profile_data) const {
  if (detail == nullptr) {
    return;
  }

  op_profile_data->set_node_type(detail->type);
  OpProfilingStat* inference_stat =
      op_profile_data->mutable_inference_microseconds();
  inference_stat->set_first(detail->elapsed_time.first());
  inference_stat->set_last(detail->elapsed_time.newest());
  inference_stat->set_avg(detail->elapsed_time.avg());
  inference_stat->set_stddev(detail->elapsed_time.std_deviation());
  inference_stat->set_variance(detail->elapsed_time.variance());
  inference_stat->set_min(detail->elapsed_time.min());
  inference_stat->set_max(detail->elapsed_time.max());
  inference_stat->set_sum(detail->elapsed_time.sum());
  inference_stat->set_count(detail->elapsed_time.count());

  OpProfilingStat* memory_stat = op_profile_data->mutable_mem_kb();
  memory_stat->set_first(detail->mem_used.first() / 1000.0);
  memory_stat->set_last(detail->mem_used.newest() / 1000.0);
  memory_stat->set_avg(detail->mem_used.avg() / 1000.0);
  memory_stat->set_stddev(detail->mem_used.std_deviation() / 1000.0);
  memory_stat->set_variance(detail->mem_used.variance() / 1000000.0);
  memory_stat->set_min(detail->mem_used.min() / 1000.0);
  memory_stat->set_max(detail->mem_used.max() / 1000.0);
  memory_stat->set_sum(detail->mem_used.sum() / 1000.0);
  memory_stat->set_count(detail->mem_used.count());

  op_profile_data->set_times_called(detail->times_called /
                                    stats_calculator->num_runs());
  op_profile_data->set_name(detail->name);
  op_profile_data->set_run_order(detail->run_order);
}

void ProfileSummaryProtoFormatter::GenerateSubGraphProfilingData(
    const tensorflow::StatsCalculator* stats_calculator, int subgraph_index,
    const std::map<uint32_t, std::string>& subgraph_name_map,
    SubGraphProfilingData* const sub_graph_profiling_data) const {
  sub_graph_profiling_data->set_subgraph_index(subgraph_index);

  std::string subgraph_name = "";
  if (subgraph_name_map.find(subgraph_index) != subgraph_name_map.end()) {
    subgraph_name = subgraph_name_map.at(subgraph_index);
  }
  sub_graph_profiling_data->set_subgraph_name(subgraph_name);

  for (tensorflow::StatsCalculator::Detail& detail :
       GetDetailsSortedByRunOrder(stats_calculator)) {
    OpProfileData* const op_profile_data =
        sub_graph_profiling_data->add_per_op_profiles();
    GenerateOpProfileDataFromDetail(&detail, stats_calculator, op_profile_data);
  }
}

void ProfileSummaryProtoFormatter::GenerateDelegateProfilingData(
    const tensorflow::StatsCalculator* stats_calculator,
    DelegateProfilingData* const delegate_profiling_data) const {
  for (const tensorflow::StatsCalculator::Detail& detail :
       GetDetailsSortedByRunOrder(stats_calculator)) {
    OpProfileData* const op_profile_data =
        delegate_profiling_data->add_per_op_profiles();
    GenerateOpProfileDataFromDetail(&detail, stats_calculator, op_profile_data);
  }
}

std::string ProfileSummaryProtoFormatter::GetShortSummary(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator,
    const std::map<uint32_t, std::string>& subgraph_name_map) const {
  TFLITE_LOG(ERROR) << "GetShortSummary is not supported for proto formatter.";
  return "";
}

std::string ProfileSummaryProtoFormatter::GetOutputString(
    const std::map<uint32_t, std::unique_ptr<tensorflow::StatsCalculator>>&
        stats_calculator_map,
    const tensorflow::StatsCalculator& delegate_stats_calculator,
    const std::map<uint32_t, std::string>& subgraph_name_map) const {
  ModelProfilingData model_profiling_data;
  for (const auto& stats_calc : stats_calculator_map) {
    auto subgraph_index = stats_calc.first;
    tensorflow::StatsCalculator* subgraph_stats = stats_calc.second.get();
    SubGraphProfilingData* const sub_graph_profiling_data =
        model_profiling_data.add_subgraph_profiles();
    GenerateSubGraphProfilingData(subgraph_stats, subgraph_index,
                                  subgraph_name_map, sub_graph_profiling_data);
  }

  if (delegate_stats_calculator.num_runs() > 0) {
    DelegateProfilingData* const delegate_profiling_data =
        model_profiling_data.add_delegate_profiles();
    GenerateDelegateProfilingData(&delegate_stats_calculator,
                                  delegate_profiling_data);
  }

  return model_profiling_data.SerializeAsString();
}

tensorflow::StatSummarizerOptions
ProfileSummaryProtoFormatter::GetStatSummarizerOptions() const {
  auto options = tensorflow::StatSummarizerOptions();
  // Summary will be manually handled per subgraphs in order to keep the
  // compatibility.
  options.show_summary = false;
  options.show_memory = false;
  return options;
}

void ProfileSummaryProtoFormatter::HandleOutput(
    const std::string& init_output, const std::string& run_output,
    std::string output_file_path) const {
  std::ofstream output_file(output_file_path, std::ios_base::binary);
  std::ostream* output_stream = nullptr;
  if (output_file.good()) {
    output_stream = &output_file;
  }

  BenchmarkProfilingData benchmark_profiling_data;
  if (!init_output.empty()) {
    benchmark_profiling_data.mutable_init_profile()->ParseFromString(
        init_output);
  }
  if (!run_output.empty()) {
    benchmark_profiling_data.mutable_runtime_profile()->ParseFromString(
        run_output);
  }

  if (output_stream == nullptr) {
    TFLITE_LOG(INFO) << benchmark_profiling_data.DebugString();
  } else {
    benchmark_profiling_data.SerializeToOstream(output_stream);
  }
}

}  // namespace profiling
}  // namespace tflite
