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
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/costmodel.h"
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

  std::unordered_map<string, DeviceStats> per_device_stats;
  std::unordered_map<int, const DeviceStepStats*> gpu_hardware_stats;

  for (int i = 0; i < step_stats_->dev_stats_size(); ++i) {
    const DeviceStepStats& device_stats = step_stats_->dev_stats(i);
    const string device_name = device_stats.device();
    const int gpu_id = ExtractGpuWithStreamAll(device_name);
    if (gpu_id >= 0) {
      // These are gpu hardware stats
      gpu_hardware_stats[gpu_id] = &device_stats;
    } else {
      // The are regular stats.
      per_device_stats[device_name] = DeviceStats{&device_stats, nullptr};
    }
  }

  for (auto& itr : per_device_stats) {
    const string& device_name = itr.first;
    const int gpu_id = ExtractGpuWithoutStream(device_name);
    if (gpu_id >= 0) {
      // Reference the gpu hardware stats in addition to the regular stats
      // for this gpu device if they're available.
      if (gpu_hardware_stats.find(gpu_id) != gpu_hardware_stats.end()) {
        itr.second.hardware_stats = gpu_hardware_stats.find(gpu_id)->second;
      }
    }
  }

  for (auto itr : device_map) {
    const string device = itr.first;
    if (per_device_stats.find(device) == per_device_stats.end()) {
      continue;
    }

    const Graph* graph = itr.second;
    CostModel* cm = cost_model_manager->FindOrCreateCostModel(graph);

    std::unordered_map<string, Node*> name_to_node;
    for (Node* n : graph->nodes()) {
      name_to_node[n->name()] = n;
    }

    const DeviceStats& dev_stats = per_device_stats.find(device)->second;

    for (int i = 0; i < dev_stats.regular_stats->node_stats_size(); ++i) {
      const NodeExecStats& stats = dev_stats.regular_stats->node_stats(i);
      const Node* node = name_to_node[stats.node_name()];
      if (node) {
        for (int i = 0; i < stats.output_size(); ++i) {
          cm->RecordMaxMemorySize(node, i, Bytes(stats.output(i)
                                                     .tensor_description()
                                                     .allocation_description()
                                                     .allocated_bytes()));
          cm->RecordAllocationId(node, i, stats.output(i)
                                              .tensor_description()
                                              .allocation_description()
                                              .allocation_id());
        }
        // Use hardware stats to record the execution time if they're available,
        // otherwise use the regular (less accurate) stats
        if (dev_stats.hardware_stats &&
            i < dev_stats.hardware_stats->node_stats_size()) {
          const NodeExecStats& hw_stats =
              dev_stats.hardware_stats->node_stats(i);
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
    if (!step_stats_) {
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
  }
  delete nt;
}

void StepStatsCollector::Swap(StepStats* ss) {
  mutex_lock l(mu_);
  CHECK(step_stats_);
  ss->Swap(step_stats_);
}

}  // namespace tensorflow
