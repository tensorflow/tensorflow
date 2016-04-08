/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

StepStatsCollector::StepStatsCollector(
    StepStats* ss, std::unordered_map<const Graph*, CostModel*>* cm)
    : step_stats_(ss), cost_models_(cm) {}

void StepStatsCollector::UpdateCostModel(const NodeExecStats* nt,
                                         const Graph* graph, const Node* node) {
  mutex_lock l(mu_);
  if (!cost_models_) {
    return;
  }
  CostModel* cm;
  auto it = cost_models_->find(graph);
  if (it == cost_models_->end()) {
    cm = new CostModel(false);
    cm->InitFromGraph(*graph);
    cost_models_->emplace(graph, cm);
  } else {
    cm = (*it).second;
  }

  cm->RecordMaxExecutionTime(node, Microseconds(nt->op_end_rel_micros()));

  for (int i = 0; i < nt->output_size(); ++i) {
    cm->RecordMaxSize(node, i, Bytes(nt->output(i)
                                         .tensor_description()
                                         .allocation_description()
                                         .allocated_bytes()));
    cm->RecordAliases(node, i, nt->output(i)
                                   .tensor_description()
                                   .allocation_description()
                                   .allocation_id());
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
