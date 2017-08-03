/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

StepStatsCollector::StepStatsCollector(StepStats* ss) : step_stats_(ss) {}

static int ExtractGpuWithStreamAll(string device_name) {
  // Check if the device name matches the ".*gpu:(\\d+)/stream:all$" regexp,
  // and if it does return the stream index (always positive). If it doesn't
  // return -1.

  // The best way to parse this regexp using a scanner is to parse it in
  // reverse starting from the end.
  std::reverse(device_name.begin(), device_name.end());
  strings::Scanner scanner(device_name);
  // Check that the string end with '/stream:all'
  scanner.OneLiteral("lla:maerts/");
  // Capture the digits if present
  scanner.RestartCapture().Many(strings::Scanner::DIGIT).StopCapture();
  // Check that the digits are preceded by the 'gpu:' string
  scanner.OneLiteral(":upg");
  StringPiece capture;
  bool matched = scanner.GetResult(nullptr, &capture);

  if (!matched) {
    return -1;
  } else {
    // Convert the captured string into an integer. But first we need to put
    // the digits back in order
    string ordered_capture = capture.ToString();
    std::reverse(ordered_capture.begin(), ordered_capture.end());
    int gpu_id;
    CHECK(strings::safe_strto32(ordered_capture, &gpu_id));
    return gpu_id;
  }
}

static int ExtractGpuWithoutStream(string device_name) {
  // Check if the device name matches the ".*gpu:(\\d+)$" regexp,
  // and if it does return the stream index (always positive). If it doesn't
  // return -1.

  // The best way to parse this regexp using a scanner is to parse it in
  // reverse starting from the end.
  std::reverse(device_name.begin(), device_name.end());
  strings::Scanner scanner(device_name);
  // Capture the trailing digits if present
  scanner.RestartCapture().Many(strings::Scanner::DIGIT).StopCapture();
  // Check that the digits are preceded by the 'gpu:' string
  scanner.OneLiteral(":upg");
  StringPiece capture;
  bool matched = scanner.GetResult(nullptr, &capture);

  if (!matched) {
    return -1;
  } else {
    // Convert the captured string into an integer. But first we need to put
    // the digits back in order
    string ordered_capture = capture.ToString();
    std::reverse(ordered_capture.begin(), ordered_capture.end());
    int gpu_id;
    CHECK(strings::safe_strto32(ordered_capture, &gpu_id));
    return gpu_id;
  }
}

void StepStatsCollector::BuildCostModel(
    CostModelManager* cost_model_manager,
    const std::unordered_map<string, const Graph*>& device_map) {
  mutex_lock lock(mu_);

  // Hardware stats for gpu are available under a fake device named
  // "gpu:<id>/stream::all.
  // Use them instead of regular stats whenever they're available to extract
  // the execution stats of a particular node since they're more accurate.
  // However hardware traces don't record memory usage, so we still have to
  // rely on regular traces to track memory usage.
  struct DeviceStats {
    const DeviceStepStats* regular_stats;
    const DeviceStepStats* hardware_stats;
  };

  std::unordered_map<StringPiece, DeviceStats, StringPiece::Hasher>
      per_device_stats;
  std::unordered_map<int, const DeviceStepStats*> gpu_hardware_stats;

  for (int i = 0; i < step_stats_->dev_stats_size(); ++i) {
    const DeviceStepStats& device_stats = step_stats_->dev_stats(i);
    const string& device_name = device_stats.device();
    const int gpu_id = ExtractGpuWithStreamAll(device_name);
    if (gpu_id >= 0) {
      // These are gpu hardware stats
      gpu_hardware_stats.emplace(gpu_id, &device_stats);
    } else {
      // These are regular stats.
      per_device_stats.emplace(device_name,
                               DeviceStats{&device_stats, nullptr});
    }
  }

  for (auto& itr : per_device_stats) {
    const StringPiece device_name = itr.first;
    const int gpu_id = ExtractGpuWithoutStream(device_name.ToString());
    if (gpu_id >= 0) {
      // Reference the gpu hardware stats in addition to the regular stats
      // for this gpu device if they're available.
      if (gpu_hardware_stats.find(gpu_id) != gpu_hardware_stats.end()) {
        itr.second.hardware_stats = gpu_hardware_stats.find(gpu_id)->second;
      }
    }
  }

  for (auto itr : device_map) {
    const StringPiece device = itr.first;
    if (per_device_stats.find(device) == per_device_stats.end()) {
      continue;
    }

    const Graph* graph = itr.second;
    CostModel* cm = cost_model_manager->FindOrCreateCostModel(graph);
    cm->IncrementUpdateTimes();

    std::unordered_map<StringPiece, Node*, StringPiece::Hasher> name_to_node;
    for (Node* n : graph->nodes()) {
      name_to_node.emplace(n->name(), n);
    }

    const DeviceStats& dev_stats = per_device_stats.find(device)->second;

    std::unordered_map<string, NodeExecStats> name_to_hw_node_stats;
    if (dev_stats.hardware_stats) {
      for (const auto& node_stats : dev_stats.hardware_stats->node_stats()) {
        string node_name = node_stats.node_name();
        // Remove the part of op name (e.g. :Conv2D) in the end of a node name.
        size_t pos = node_name.find_first_of(":");
        if (pos != std::string::npos) {
          node_name = node_name.substr(0, pos);
        }
        // Certain ops (e.g. Conv2D) are implemented with multiple GPU kernels,
        // which results in multiple NodeExecStats with the same node name. For
        // such ops, we sum up the time for all its GPU kernels.
        if (name_to_hw_node_stats.find(node_name) !=
            name_to_hw_node_stats.end()) {
          int64 time = name_to_hw_node_stats[node_name].op_end_rel_micros();
          name_to_hw_node_stats[node_name].set_op_end_rel_micros(
              time + node_stats.op_end_rel_micros());
        } else {
          name_to_hw_node_stats.emplace(node_name, node_stats);
        }
      }
    }

    for (int i = 0; i < dev_stats.regular_stats->node_stats_size(); ++i) {
      const NodeExecStats& stats = dev_stats.regular_stats->node_stats(i);
      const Node* node = name_to_node[stats.node_name()];
      if (node) {
        for (int i = 0; i < stats.output_size(); ++i) {
          const auto& output = stats.output(i);
          cm->RecordMaxMemorySize(node, i, Bytes(output.tensor_description()
                                                     .allocation_description()
                                                     .allocated_bytes()),
                                  stats.output(i).tensor_description().shape(),
                                  node->output_types()[i]);
          cm->RecordAllocationId(node, i, output.tensor_description()
                                              .allocation_description()
                                              .allocation_id());
        }
        cm->RecordMemoryStats(node, stats.memory_stats());
        // Use hardware stats to record the execution time if they're available,
        // otherwise use the regular (less accurate) stats
        string node_name = dev_stats.regular_stats->node_stats(i).node_name();
        if (dev_stats.hardware_stats &&
            name_to_hw_node_stats.find(node_name) !=
                name_to_hw_node_stats.end()) {
          const NodeExecStats& hw_stats = name_to_hw_node_stats[node_name];
          cm->RecordMaxExecutionTime(
              node, Microseconds(hw_stats.op_end_rel_micros()));
        } else {
          cm->RecordMaxExecutionTime(node,
                                     Microseconds(stats.op_end_rel_micros()));
        }
      }
    }
  }
}

void StepStatsCollector::Save(const string& device, NodeExecStats* nt) {
  VLOG(1) << "Save dev " << device << " nt " << nt;
  {
    mutex_lock l(mu_);
    if (!step_stats_ || collectedNodes >= kMaxCollectedNodes) {
      VLOG(1) << "step_stats_ nullptr or already collected too many nodes.";
      delete nt;
      return;
    }
    DeviceStepStats* dss = nullptr;
    // Slow linear scan, but it should only be called
    // by a Worker in a context with < ~10 devices.
    // TODO(tucker): consider adding a std::unordered_map.
    for (auto& ds : *step_stats_->mutable_dev_stats()) {
      if (ds.device() == device) {
        dss = &ds;
        break;
      }
    }
    if (dss == nullptr) {
      dss = step_stats_->add_dev_stats();
      dss->set_device(device);
    }
    nt->Swap(dss->add_node_stats());
    collectedNodes++;
  }
  delete nt;
}

void StepStatsCollector::Swap(StepStats* ss) {
  mutex_lock l(mu_);
  CHECK(step_stats_);
  ss->Swap(step_stats_);
  collectedNodes = 0;
}

}  // namespace tensorflow
