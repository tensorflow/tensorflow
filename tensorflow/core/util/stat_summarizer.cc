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
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using Detail = StatsCalculator::Detail;

StatSummarizer::StatSummarizer(const StatSummarizerOptions& options)
    : stats_calculator_(new StatsCalculator(options)) {}

StatSummarizer::StatSummarizer(const tensorflow::GraphDef& tensorflow_graph)
    : stats_calculator_(new StatsCalculator(StatSummarizerOptions())) {}

StatSummarizer::~StatSummarizer() {}

void StatSummarizer::Validate(const std::vector<TensorDescription>* outputs,
                              const NodeExecStats& ns) const {
  if (outputs->size() != ns.output_size()) {
    LOG(WARNING) << "Number of outputs changed between runs for '"
                 << ns.node_name() << "' - was " << outputs->size() << ", now "
                 << ns.output_size();
  } else {
    for (const auto& output : ns.output()) {
      const int32_t slot = output.slot();
      if ((slot < 0) || (slot >= ns.output_size())) {
        // This is not a hard error for Switch ops, so just pass.
        continue;
      }
      const auto& stored = (*outputs)[slot];
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

void StatSummarizer::PrintStepStats() const {
  string output = GetOutputString();
  std::istringstream iss(output);
  for (std::string line; std::getline(iss, line);) {
    LOG(INFO) << line;
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
  std::string::size_type end = label.find('(', start);
  if (end == std::string::npos) return "<>";
  return label.substr(start, end - start);
}
}  // namespace

void StatSummarizer::ProcessStepStats(const StepStats& step_stats) {
  int64_t curr_total_us = 0;
  int64_t mem_total = 0;

  int64_t first_node_start_us =
      (step_stats.dev_stats_size() > 0 &&
       step_stats.dev_stats(0).node_stats_size() > 0)
          ? step_stats.dev_stats(0).node_stats(0).all_start_micros()
          : 0;

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
      // NOTE(fishx): We will record ops execution time twice: one as CPU
      // activity with device name "/host:CPU" and the other as TF runtime
      // activity with device name started with "/job:*". It is safe to ignore
      // CPU activities here.
      // TODO(b/138729463): Read ops execution time from CPU activities instead
      // of runtime activities.
      if (ds.device().find("/host:CPU") != std::string::npos) {
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
      const int64_t curr_time = ns.all_end_rel_micros();
      curr_total_us += curr_time;
      auto output_result =
          outputs_.emplace(name, std::vector<TensorDescription>());
      std::vector<TensorDescription>* outputs = &(output_result.first->second);

      int64_t start_us = (ns.all_start_micros() - first_node_start_us);
      int64_t rel_end_us = curr_time;

      // If this is the first pass, initialize some values.
      if (output_result.second) {
        outputs->resize(ns.output_size());
        for (const auto& output : ns.output()) {
          const int32_t slot = output.slot();
          if ((slot < 0) || (slot >= ns.output_size())) {
            // This is not a hard error for Switch ops, so just pass.
            continue;
          }
          (*outputs)[slot] = output.tensor_description();
        }
      }

      int64_t curr_node_mem = 0;
      for (const auto& mem : ns.memory()) {
        const int64_t mem_usage = mem.total_bytes();
        curr_node_mem += mem_usage;
      }
      stats_calculator_->AddNodeStats(name, op_type, node_num, start_us,
                                      rel_end_us, curr_node_mem);

      mem_total += curr_node_mem;

      Validate(outputs, ns);
    }
  }

  stats_calculator_->UpdateRunTotalUs(curr_total_us);
  stats_calculator_->UpdateMemoryUsed(mem_total);
}


void StatSummarizer::PrintOutputs() const {
  std::priority_queue<
      std::pair<int64_t, const std::pair<const std::string, Detail>*>>
      timings;
  for (const auto& entry : stats_calculator_->GetDetails()) {
    timings.emplace(-entry.second.start_us.avg(), &entry);
  }

  LOG(INFO) << "============ Node output tensor sizes in run order ========";
  while (!timings.empty()) {
    auto entry = timings.top();
    timings.pop();
    std::stringstream stream;
    const auto detail_outputs = outputs_.at(entry.second->first);
    stream << entry.second->first << "\t" << detail_outputs.size();
    for (const auto& tensor : detail_outputs) {
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
