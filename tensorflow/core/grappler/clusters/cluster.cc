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
#include <atomic>

namespace tensorflow {
namespace grappler {

static std::atomic<bool> already_created(false);

Cluster::Cluster(int timeout_s) : timeout_s_(timeout_s) {
  // This is really ugly: to avoid leaking variables, we need to reset the tf
  // session every time we're done processing a grappler item. However,
  // variables are global, and therefore we can't have more than 1 session alive
  // at a time. This check detects when more that one cluster is created.
  CHECK(!already_created);
  already_created = true;

  DisableDetailedStats(false);
}

Cluster::~Cluster() {
  CHECK(already_created);
  already_created = false;
}

void Cluster::AllowSoftPlacement(bool soft_placement_state) {
  options_.config.set_allow_soft_placement(soft_placement_state);
}

void Cluster::SetNumWarmupSteps(int num_steps) {
  options_.config.mutable_graph_options()->set_build_cost_model_after(
      num_steps);
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

}  // end namespace grappler
}  // end namespace tensorflow
