/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

Cluster::Cluster(int timeout_s) : timeout_s_(timeout_s) {
  DisableDetailedStats(false);
}

Cluster::~Cluster() {}

void Cluster::AllowSoftPlacement(bool soft_placement_state) {
  options_.config.set_allow_soft_placement(soft_placement_state);
}

void Cluster::SetNumWarmupSteps(int num_steps) {
  options_.config.mutable_graph_options()->set_build_cost_model_after(
      num_steps);
}

int Cluster::NumWarmupSteps() const {
  return options_.config.graph_options().build_cost_model_after();
}

void Cluster::DisableDetailedStats(bool disable) {
  if (disable) {
    options_.config.mutable_graph_options()->set_build_cost_model(0);
    run_options_.set_trace_level(RunOptions::NO_TRACE);
  } else {
    options_.config.mutable_graph_options()->set_build_cost_model(1);
    run_options_.set_trace_level(RunOptions::HARDWARE_TRACE);
  }
}

bool Cluster::DetailedStatsEnabled() const {
  return options_.config.graph_options().build_cost_model() != 0;
}

void Cluster::DisableOptimizer(bool disable) {
  OptimizerOptions* options =
      options_.config.mutable_graph_options()->mutable_optimizer_options();
  if (disable) {
    options->set_opt_level(OptimizerOptions::L0);
    // Disable Grappler optimizations.
    auto rewriter_config =
        options_.config.mutable_graph_options()->mutable_rewrite_options();
    rewriter_config->set_layout_optimizer(RewriterConfig::OFF);
    rewriter_config->set_disable_model_pruning(true);
    rewriter_config->set_constant_folding(RewriterConfig::OFF);
    rewriter_config->set_memory_optimization(RewriterConfig::NO_MEM_OPT);
    rewriter_config->mutable_auto_parallel()->set_enable(false);
    rewriter_config->clear_optimizers();
  } else {
    options->set_opt_level(OptimizerOptions::L1);
    auto rewriter_config =
        options_.config.mutable_graph_options()->mutable_rewrite_options();
    rewriter_config->set_constant_folding(RewriterConfig::DEFAULT);
    rewriter_config->set_memory_optimization(RewriterConfig::DEFAULT_MEM_OPT);
  }
}

const std::vector<string> Cluster::GetDeviceNames() const {
  std::vector<string> device_names;
  device_names.reserve(devices_.size());
  for (const auto& device : devices_) {
    device_names.push_back(device.first);
  }
  std::sort(device_names.begin(), device_names.end());
  return device_names;
}

}  // end namespace grappler
}  // end namespace tensorflow
