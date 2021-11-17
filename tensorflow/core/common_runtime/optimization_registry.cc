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

#include "tensorflow/core/common_runtime/optimization_registry.h"

#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

// static
OptimizationPassRegistry* OptimizationPassRegistry::Global() {
  static OptimizationPassRegistry* global_optimization_registry =
      new OptimizationPassRegistry;
  return global_optimization_registry;
}

void OptimizationPassRegistry::Register(
    Grouping grouping, int phase, std::unique_ptr<GraphOptimizationPass> pass) {
  groups_[grouping][phase].push_back(std::move(pass));
}

Status OptimizationPassRegistry::RunGrouping(
    Grouping grouping, const GraphOptimizationPassOptions& options) {
  auto dump_graph = [&](std::string& prefix) {
    if (options.graph) {
      DumpGraphToFile(
          strings::StrCat(prefix, "_",
                          reinterpret_cast<uintptr_t>((*options.graph).get())),
          **options.graph, options.flib_def);
    }
    if (options.partition_graphs) {
      for (auto& part : *options.partition_graphs) {
        DumpGraphToFile(
            strings::StrCat(prefix, "_partition_", part.first, "_",
                            reinterpret_cast<uintptr_t>(part.second.get())),
            *part.second, options.flib_def);
      }
    }
  };

  VLOG(1) << "Starting optimization of a group " << grouping;
  if (VLOG_IS_ON(2)) {
    std::string prefix = strings::StrCat("before_grouping_", grouping);
    dump_graph(prefix);
  }
  auto group = groups_.find(grouping);
  if (group != groups_.end()) {
    static const char* kGraphOptimizationCategory = "GraphOptimizationPass";
    tensorflow::metrics::ScopedCounter<2> group_timings(
        tensorflow::metrics::GetGraphOptimizationCounter(),
        {kGraphOptimizationCategory, "*"});
    for (auto& phase : group->second) {
      VLOG(1) << "Running optimization phase " << phase.first;
      for (auto& pass : phase.second) {
        VLOG(1) << "Running optimization pass: " << pass->name();

        tensorflow::metrics::ScopedCounter<2> pass_timings(
            tensorflow::metrics::GetGraphOptimizationCounter(),
            {kGraphOptimizationCategory, pass->name()});
        Status s = pass->Run(options);

        if (!s.ok()) return s;
        pass_timings.ReportAndStop();
        if (VLOG_IS_ON(5)) {
          std::string prefix =
              strings::StrCat("after_group_", grouping, "_phase_", phase.first,
                              "_", pass->name());
          dump_graph(prefix);
        }
      }
    }
    group_timings.ReportAndStop();
  }
  VLOG(1) << "Finished optimization of a group " << grouping;
  if (VLOG_IS_ON(2)) {
    std::string prefix = strings::StrCat("after_grouping_", grouping);
    dump_graph(prefix);
  }
  return Status::OK();
}

void OptimizationPassRegistry::LogGrouping(Grouping grouping, int vlog_level) {
  auto group = groups_.find(grouping);
  if (group != groups_.end()) {
    for (auto& phase : group->second) {
      for (auto& pass : phase.second) {
        VLOG(vlog_level) << "Registered optimization pass grouping " << grouping
                         << " phase " << phase.first << ": " << pass->name();
      }
    }
  }
}

void OptimizationPassRegistry::LogAllGroupings(int vlog_level) {
  for (auto group = groups_.begin(); group != groups_.end(); ++group) {
    LogGrouping(group->first, vlog_level);
  }
}

}  // namespace tensorflow
