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

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

StatSummarizer::StatSummarizer(const StatSummarizerOptions& options)
    : options_(options) {}

StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph)
    : StatSummarizer(StatSummarizerOptions()) {}

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
        // This is not a hard error for Switch ops, so just pass.
        continue;
      }
      const auto& stored = detail->outputs[slot];
      const auto& current = output.tensor_description();

      bool do_tensors_match =
          (stored.dtype() == current.dtype()) &&
          (stored.shape().dim_size() == current.shape().dim_size());

      if (do_tensors_match) {
        for (int i = 0; i < stored.shape().dim_size(); ++i) {
          if (stored.shape().dim(i).size() != current.shape().dim(i).size()) {
            do_tensors_match = false;
            break;
          }
        }
      }

      if (!do_tensors_match) {
        LOG(WARNING) << "Output tensor changed between runs for '"
                     << ns.node_name();
      }
    }
  }
}

namespace {
std::string OpType(const DeviceStepStats& ds, const NodeExecStats& ns) {
  // There is no published specification of how DeviceStats and NodeStats
  // are filled in. Thus, we live with the fragility of this implementation.
  //
  // Note that NodeStats.node_name may NOT refer to a node in the Graph.
  // This can happen if, either:
  // (1) The DeviceStats corresponds to statistics from the GPUTracer
  //     logging (which adds devices whose name contains either "/stream"
  //     or "/memcpy" to the StepStats), OR
  // (2) The graph was partitioned, and thus the NodeStats refers to
  //     the SendTensor or RecvTensor operations added.
  // For these cases, return "<>" as the "type" of the operation.
  //
  // The StatSummarizer was initially aimed at CPU execution on mobile, where
  // there was no GPUTracing and no graph partitioning, so the conditions above
  // do not occur.
  //
  // It would be nice to have a clearer spec for StepStats so utilities such as
  // this class can handle nodes that do not appear in the original graph
  // gracefully. Till then, duplicate what is done by:
  // https://www.tensorflow.org/code/tensorflow/python/client/timeline.py
  // and rely on the unittest.
  if (ds.device().find("/stream") != std::string::npos ||
      ds.device().find("/memcpy") != std::string::npos) {
    // Stats from the GPUTracer, does not correspond to TensorFlow ops.
    return "<>";
  }
  // timeline_label should be of the format: <node_name> = <op_type>(<args>)
  // Extract <op_type>.
  const std::string sep(" = ");
  const std::string& label = ns.timeline_label();
  std::string::size_type start = label.find(sep);
  if (start == std::string::npos) return "<>";
  start += sep.size();
  std::string::size_type end = label.find("(", start);
  if (end == std::string::npos) return "<>";
  return label.substr(start, end - start);
}
}  // namespace

void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  int64 curr_total_us = 0;
  int64 mem_total = 0;

  int64 first_node_start_us =
      step_stats.dev_stats(0).node_stats(0).all_start_micros();

  int node_num = 0;
  for (const auto& ds : step_stats.dev_stats()) {
    for (const auto& ns : ds.node_stats()) {
      // NOTE(blackhc): To better support GPUs:
      // GPU kernels are duplicated both in /stream:all and their
      // /stream:$index. GPU memcpys are duplicated both in /memcpy and their
      // /stream:$index. So only keep /stream:all and /memcpy and ignore all
      // /stream:$index to only count GPU executions once.
      if (ds.device().find("/stream") != std::string::npos &&
          ds.device().find("/stream:all") == std::string::npos) {
        continue;
      }

      std::string name = ns.node_name();
      std::string op_type = "<>";
      // NOTE(blackhc): we have to ensure that all keys into the detail map
      // are unique, so we add [Kernel] or [MemCpy] as a suffix to the name.
      // To make the node type summary work better, we prefix "gpu:" to
      // the op type when the info is from a /gpu/stream or /memcpy channel.
      if (ds.device().find("/stream") != std::string::npos) {
        // node_name: name ":" opType
        auto parts = str_util::Split(ns.node_name(), ':');
        if (parts.size() == 2) {
          name = parts[0] + " [Kernel]";
          op_type = "gpu:" + parts[1];
        }
      } else if (ds.device().find("/memcpy") != std::string::npos) {
        // node_name: name (":" opType)? ":" memCpyType
        auto parts = str_util::Split(ns.node_name(), ':');
        if (parts.size() == 2 || parts.size() == 3) {
          name = parts.front() + " [MemCpy]";
          // We don't care about the actual op type (it might not be available
          // for edge_ memcpys). We only care that it's a memcpy for now.
          op_type = "gpu:" + parts.back();
        }
      } else {
        op_type = OpType(ds, ns);
      }

      ++node_num;
      const int64 curr_time = ns.all_end_rel_micros();
      curr_total_us += curr_time;
      auto result = details_.emplace(name, Detail());
      Detail* detail = &(result.first->second);

      detail->start_us.UpdateStat(ns.all_start_micros() - first_node_start_us);
      detail->rel_end_us.UpdateStat(curr_time);

      // If this is the first pass, initialize some values.
      if (result.second) {
        detail->name = name;
        detail->type = op_type;

        detail->run_order = node_num;

        detail->outputs.resize(ns.output_size());
        for (const auto& output : ns.output()) {
          const int32 slot = output.slot();
          if ((slot < 0) || (slot >= ns.output_size())) {
            // This is not a hard error for Switch ops, so just pass.
            continue;
          }
          detail->outputs[slot] = output.tensor_description();
        }

        detail->times_called = 0;
      }

      int64 curr_node_mem = 0;
      for (const auto& mem : ns.memory()) {
        const int64 mem_usage = mem.total_bytes();
        curr_node_mem += mem_usage;
      }
      detail->mem_used.UpdateStat(curr_node_mem);
      mem_total += curr_node_mem;

      ++detail->times_called;

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

  stream << details_.size() << " nodes observed" << std::endl;
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
  InitField(stream, 9) << "[times called]";
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
  const int64 times_called = detail.times_called / num_runs();

  std::stringstream stream;
  InitField(stream, 24) << detail.type;
  InitField(stream, 9) << start_ms;
  InitField(stream, 9) << first_time_ms;
  InitField(stream, 9) << avg_time_ms;
  InitField(stream, 7) << percentage << "%";
  InitField(stream, 7) << cdf_percentage << "%";
  InitField(stream, 10) << detail.mem_used.newest() / 1000.0;
  InitField(stream, 9) << times_called;
  stream << "\t" << detail.name;

  return stream.str();
}

void StatSummarizer::OrderNodesByMetric(
    SortingMetric metric, std::vector<const Detail*>* details) const {
  std::priority_queue<std::pair<string, const Detail*>> sorted_list;
  const int num_nodes = details_.size();

  for (const auto& det : details_) {
    const Detail* detail = &(det.second);
    std::stringstream stream;
    stream << std::setw(20) << std::right << std::setprecision(10)
           << std::fixed;

    switch (metric) {
      case BY_NAME:
        stream << detail->name;
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

void StatSummarizer::ComputeStatsByType(
    std::map<string, int64>* node_type_map_count,
    std::map<string, int64>* node_type_map_time,
    std::map<string, int64>* node_type_map_memory,
    std::map<string, int64>* node_type_map_times_called,
    int64* accumulated_us) const {
  int64 run_count = run_total_us_.count();

  for (const auto& det : details_) {
    const string node_name = det.first;
    const Detail& detail = det.second;

    int64 curr_time_val =
        static_cast<int64>(detail.rel_end_us.sum() / run_count);
    *accumulated_us += curr_time_val;

    int64 curr_memory_val = detail.mem_used.newest();

    const string& node_type = detail.type;

    (*node_type_map_count)[node_type] += 1;
    (*node_type_map_time)[node_type] += curr_time_val;
    (*node_type_map_memory)[node_type] += curr_memory_val;
    (*node_type_map_times_called)[node_type] += detail.times_called / run_count;
  }
}

std::string StatSummarizer::GetStatsByNodeType() const {
  std::stringstream stream;

  stream << "============================== Summary by node type "
            "=============================="
         << std::endl;

  LOG(INFO) << "Number of nodes executed: " << details_.size();

  std::map<string, int64> node_type_map_count;
  std::map<string, int64> node_type_map_time;
  std::map<string, int64> node_type_map_memory;
  std::map<string, int64> node_type_map_times_called;
  int64 accumulated_us = 0;

  ComputeStatsByType(&node_type_map_count, &node_type_map_time,
                     &node_type_map_memory, &node_type_map_times_called,
                     &accumulated_us);

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
  InitField(stream, 10) << "[times called]";
  stream << std::endl;

  float cdf = 0.0f;
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();

    const string node_type = entry.second.first;
    const float memory = entry.second.second / 1000.0f;

    const int64 node_type_total_us = entry.first;
    const float time_per_run_ms = node_type_total_us / 1000.0f;

    const float percentage =
        ((entry.first / static_cast<float>(accumulated_us)) * 100.0f);
    cdf += percentage;

    InitField(stream, 24) << node_type;
    InitField(stream, 9) << node_type_map_count[node_type];
    InitField(stream, 10) << time_per_run_ms;
    InitField(stream, 10) << percentage << "%";
    InitField(stream, 10) << cdf << "%";
    InitField(stream, 10) << memory;
    InitField(stream, 9) << node_type_map_times_called[node_type];
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
