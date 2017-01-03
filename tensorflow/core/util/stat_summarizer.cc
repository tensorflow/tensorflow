/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/stat_summarizer.h"

#include <iomanip>
#include <map>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph)
    : StatSummarizer(tensorflow_graph, StatSummarizerOptions()) {}

StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph,
                               const StatSummarizerOptions& options)
    : options_(options) {
  LOG(INFO) << "StatSummarizer found " << tensorflow_graph.node_size()
            << " nodes";
  for (const auto& node : tensorflow_graph.node()) {
    nodes_in_def_order_.push_back(node.name());
    node_types_[node.name()] = node.op();
  }
}

void StatSummarizer::Validate(const Detail* detail,
                              const NodeExecStats& ns) const {
  if (detail->outputs.size() != ns.output_size()) {
    LOG(WARNING) << "Number of outputs changed between runs for '"
                 << ns.node_name() << "' - was " << detail->outputs.size()
                 << ", now " << ns.output_size();
  } else {
    for (const auto& output : ns.output()) {
      const int32 slot = output.slot();
      if ((slot < 0) || (slot >= ns.output_size())) {
        LOG(ERROR) << "Bad output slot '" << slot << "' for '" << ns.node_name()
                   << "'";
        return;
      }
      const auto& stored = detail->outputs[slot];
      const auto& current = output.tensor_description();
      bool do_shapes_match = true;
      if (stored.shape().dim_size() != current.shape().dim_size()) {
        do_shapes_match = false;
      } else {
        for (int i = 0; i < stored.shape().dim_size(); ++i) {
          if (stored.shape().dim(i).size() != current.shape().dim(i).size()) {
            do_shapes_match = false;
          }
        }

        if ((stored.dtype() != current.dtype()) || !do_shapes_match) {
          LOG(WARNING) << "Output tensor changed between runs for '"
                       << ns.node_name();
        }
      }
    }
  }
}

void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  int64 curr_total_us = 0;
  int64 mem_total = 0;

  int64 first_node_start_us =
      step_stats.dev_stats(0).node_stats(0).all_start_micros();

  int node_num = 0;
  for (const auto& ds : step_stats.dev_stats()) {
    for (const auto& ns : ds.node_stats()) {
      ++node_num;
      const int64 curr_time = ns.all_end_rel_micros();
      curr_total_us += curr_time;
      auto result = details_.emplace(ns.node_name(), Detail());
      Detail* detail = &(result.first->second);

      detail->start_us.UpdateStat(ns.all_start_micros() - first_node_start_us);
      detail->rel_end_us.UpdateStat(curr_time);

      // If this is the first pass, initialize some values.
      if (result.second) {
        detail->name = ns.node_name();

        auto node_type_it = node_types_.find(detail->name);
        if (node_type_it != node_types_.end()) {
          detail->type = node_type_it->second;
        }

        detail->run_order = node_num;

        detail->outputs.resize(ns.output_size());
        for (const auto& output : ns.output()) {
          const int32 slot = output.slot();
          if ((slot < 0) || (slot >= ns.output_size())) {
            LOG(ERROR) << "Bad output slot '" << slot << "' for '"
                       << ns.node_name() << "'";
            continue;
          }
          detail->outputs[slot] = output.tensor_description();
        }
      }

      int64 curr_node_mem = 0;
      for (const auto& mem : ns.memory()) {
        const int64 mem_usage = mem.total_bytes();
        curr_node_mem += mem_usage;
      }
      detail->mem_used.UpdateStat(curr_node_mem);
      mem_total += curr_node_mem;

      Validate(detail, ns);
    }
  }

  run_total_us_.UpdateStat(curr_total_us);
  memory_.UpdateStat(mem_total);
}

std::string StatSummarizer::ShortSummary() const {
  std::stringstream stream;
  stream << "Timings (microseconds): ";
  run_total_us_.OutputToStream(&stream);
  stream << std::endl;

  stream << "Memory (bytes): ";
  memory_.OutputToStream(&stream);
  stream << std::endl;

  stream << node_types_.size() << " nodes defined " << details_.size()
         << " nodes observed" << std::endl;
  return stream.str();
}

std::ostream& InitField(std::ostream& stream, int width) {
  stream << "\t" << std::right << std::setw(width) << std::fixed
         << std::setprecision(3);
  return stream;
}

std::string StatSummarizer::HeaderString(const string& title) const {
  std::stringstream stream;

  stream << "============================== " << title
         << " ==============================" << std::endl;

  InitField(stream, 24) << "[node type]";
  InitField(stream, 9) << "[start]";
  InitField(stream, 9) << "[first]";
  InitField(stream, 9) << "[avg ms]";
  InitField(stream, 8) << "[%]";
  InitField(stream, 8) << "[cdf%]";
  InitField(stream, 10) << "[mem KB]";
  stream << "\t"
         << "[Name]";
  return stream.str();
}

std::string StatSummarizer::ColumnString(const Detail& detail,
                                         const int64 cumulative_stat_on_node,
                                         const Stat<int64>& stat) const {
  const double start_ms = detail.start_us.avg() / 1000.0;
  const double first_time_ms = detail.rel_end_us.first() / 1000.0;
  const double avg_time_ms = detail.rel_end_us.avg() / 1000.0;
  const double percentage = detail.rel_end_us.sum() * 100.0 / stat.sum();
  const double cdf_percentage = (cumulative_stat_on_node * 100.0f) / stat.sum();

  std::stringstream stream;
  InitField(stream, 24) << detail.type;
  InitField(stream, 9) << start_ms;
  InitField(stream, 9) << first_time_ms;
  InitField(stream, 9) << avg_time_ms;
  InitField(stream, 7) << percentage << "%";
  InitField(stream, 7) << cdf_percentage << "%";
  InitField(stream, 10) << detail.mem_used.newest() / 1000.0;
  stream << "\t" << detail.name;

  return stream.str();
}

void StatSummarizer::OrderNodesByMetric(
    SortingMetric metric, std::vector<const Detail*>* details) const {
  std::priority_queue<std::pair<string, const Detail*>> sorted_list;
  const int num_nodes = nodes_in_def_order_.size();

  for (const auto& det : details_) {
    const Detail* detail = &(det.second);
    std::stringstream stream;
    stream << std::setw(20) << std::right << std::setprecision(10)
           << std::fixed;

    int definition_index = 0;
    auto it = std::find(nodes_in_def_order_.begin(), nodes_in_def_order_.end(),
                        detail->name);
    if (it != nodes_in_def_order_.end()) {
      definition_index = std::distance(nodes_in_def_order_.begin(), it);
    }

    switch (metric) {
      case BY_NAME:
        stream << detail->name;
        break;
      case BY_DEFINITION_ORDER:
        stream << num_nodes - definition_index;
        break;
      case BY_RUN_ORDER:
        stream << num_nodes - detail->run_order;
        break;
      case BY_TIME:
        stream << detail->rel_end_us.avg();
        break;
      case BY_MEMORY:
        stream << detail->mem_used.avg();
        break;
      case BY_TYPE:
        stream << detail->type;
        break;
      default:
        stream << "";
        break;
    }

    sorted_list.emplace(stream.str(), detail);
  }

  while (!sorted_list.empty()) {
    auto entry = sorted_list.top();
    sorted_list.pop();
    details->push_back(entry.second);
  }
}

std::string StatSummarizer::GetStatsByNodeType() const {
  std::stringstream stream;

  stream << "============================== Summary by node type "
            "=============================="
         << std::endl;

  int64 accumulated_us = 0;
  int64 accumulated_bytes = 0;
  std::map<string, int64> node_type_map_count;
  std::map<string, int64> node_type_map_time;
  std::map<string, int64> node_type_map_memory;

  int64 num_processed = 0;

  LOG(INFO) << "nodes_in_def_order_ size: " << nodes_in_def_order_.size();
  LOG(INFO) << "timing_details_ size: " << details_.size();
  for (const auto& det : details_) {
    const string node_name = det.first;
    const Detail& detail = det.second;

    int64 curr_time_val = detail.rel_end_us.avg();
    accumulated_us += curr_time_val;

    ++num_processed;
    int64 curr_memory_val = detail.mem_used.newest();
    accumulated_bytes += curr_memory_val;

    string node_type = "<>";

    auto node_type_it = node_types_.find(node_name);
    if (node_type_it != node_types_.end()) {
      node_type = node_type_it->second;
    }

    node_type_map_count[node_type] += 1;
    node_type_map_time[node_type] += curr_time_val;
    node_type_map_memory[node_type] += curr_memory_val;
  }

  LOG(INFO) << "Processed " << num_processed << " nodes";

  // Sort them.
  std::priority_queue<std::pair<int64, std::pair<string, int64>>> timings;
  for (const auto& node_type : node_type_map_time) {
    const int64 mem_used = node_type_map_memory[node_type.first];
    timings.emplace(node_type.second,
                    std::pair<string, int64>(node_type.first, mem_used));
  }

  InitField(stream, 24) << "[Node type]";
  InitField(stream, 9) << "[count]";
  InitField(stream, 10) << "[avg ms]";
  InitField(stream, 11) << "[avg %]";
  InitField(stream, 11) << "[cdf %]";
  InitField(stream, 10) << "[mem KB]";
  stream << std::endl;

  float avg_total_time_ms = 0.0f;
  float cdf = 0.0f;
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();

    const string node_type = entry.second.first;
    const float memory = entry.second.second / 1000.0f;

    const int64 node_type_total_us = entry.first;
    const float time_per_run_ms = node_type_total_us / 1000.0f;

    avg_total_time_ms += time_per_run_ms;
    const float percentage =
        ((entry.first / static_cast<float>(accumulated_us)) * 100.0f);
    cdf += percentage;

    InitField(stream, 24) << node_type;
    InitField(stream, 9) << node_type_map_count[node_type];
    InitField(stream, 10) << time_per_run_ms;
    InitField(stream, 10) << percentage << "%";
    InitField(stream, 10) << cdf << "%";
    InitField(stream, 10) << memory;
    stream << std::endl;
  }
  stream << std::endl;
  return stream.str();
}

std::string StatSummarizer::GetStatsByMetric(const string& title,
                                             SortingMetric sorting_metric,
                                             int num_stats) const {
  std::vector<const Detail*> details;
  OrderNodesByMetric(sorting_metric, &details);

  double cumulative_stat_on_node = 0;

  std::stringstream stream;
  stream << HeaderString(title) << std::endl;
  int stat_num = 0;
  for (auto detail : details) {
    ++stat_num;
    if (num_stats > 0 && stat_num > num_stats) {
      break;
    }

    // TODO(andrewharp): Make this keep track of the particular metric for cdf.
    cumulative_stat_on_node += detail->rel_end_us.sum();
    stream << ColumnString(*detail, cumulative_stat_on_node, run_total_us_)
           << std::endl;
  }
  stream << std::endl;
  return stream.str();
}

std::string StatSummarizer::GetOutputString() const {
  std::stringstream stream;
  if (options_.show_run_order) {
    stream << GetStatsByMetric("Run Order", BY_RUN_ORDER,
                               options_.run_order_limit);
  }
  if (options_.show_time) {
    stream << GetStatsByMetric("Top by Computation Time", BY_TIME,
                               options_.time_limit);
  }
  if (options_.show_memory) {
    stream << GetStatsByMetric("Top by Memory Use", BY_MEMORY,
                               options_.memory_limit);
  }
  if (options_.show_type) {
    stream << GetStatsByNodeType();
  }
  if (options_.show_summary) {
    stream << ShortSummary() << std::endl;
  }
  return stream.str();
}

void StatSummarizer::PrintStepStats() const {
  string output = GetOutputString();
  std::istringstream iss(output);
  for (std::string line; std::getline(iss, line);) {
    LOG(INFO) << line;
  }
}

void StatSummarizer::PrintOutputs() const {
  std::priority_queue<
      std::pair<int64, const std::pair<const std::string, Detail>*>>
      timings;
  for (const auto& entry : details_) {
    timings.emplace(-entry.second.start_us.avg(), &entry);
  }

  LOG(INFO) << "============ Node output tensor sizes in run order ========";
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();
    const Detail& detail = entry.second->second;
    std::stringstream stream;
    stream << entry.second->first << "\t" << detail.outputs.size();
    for (const auto& tensor : detail.outputs) {
      stream << "\t" << DataTypeString(tensor.dtype());
      stream << "\t" << tensor.shape().dim_size();
      for (const auto& d : tensor.shape().dim()) {
        stream << "\t" << d.size();
      }
    }
    LOG(INFO) << stream.str();
  }
}

}  // namespace tensorflow
