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

#include <string>

#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/util/debug_data_dumper.h"
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
  const char* grouping_name = GetGroupingName(grouping);

  auto dump_graph = [&](std::string func_name, const std::string& group,
                        const std::string& tag, bool bypass_filter) {
    if (func_name.empty()) func_name = "unknown_graph";

    if (options.graph) {
      DEBUG_DATA_DUMPER()->DumpGraph(func_name, group, tag,
                                     options.graph->get(), options.flib_def,
                                     bypass_filter);
    }
    if (options.partition_graphs) {
      for (auto& part : *options.partition_graphs) {
        DEBUG_DATA_DUMPER()->DumpGraph(func_name + "_partition_" + part.first,
                                       group, tag, part.second.get(),
                                       options.flib_def, bypass_filter);
      }
    }
  };

  dump_graph(options.debug_filename_prefix, kDebugGroupMain,
             strings::StrCat("before_opt_group_", grouping_name),
             VLOG_IS_ON(3));

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
        if (options.graph) {
          VLOG(1) << "Graph #nodes " << (*options.graph)->num_nodes()
                  << " #edges " << (*options.graph)->num_edges();
        }
        tensorflow::metrics::ScopedCounter<2> pass_timings(
            tensorflow::metrics::GetGraphOptimizationCounter(),
            {kGraphOptimizationCategory, pass->name()});
        Status s = pass->Run(options);

        if (!s.ok()) return s;
        pass_timings.ReportAndStop();

        dump_graph(options.debug_filename_prefix, kDebugGroupGraphOptPass,
                   strings::StrCat("after_opt_group_", grouping_name, "_phase_",
                                   phase.first, "_", pass->name()),
                   VLOG_IS_ON(5));
      }
    }
    group_timings.ReportAndStop();
  }

  VLOG(1) << "Finished optimization of a group " << grouping;
  if (options.graph && group != groups_.end()) {
    VLOG(1) << "Graph #nodes " << (*options.graph)->num_nodes() << " #edges "
            << (*options.graph)->num_edges();
  }

  dump_graph(options.debug_filename_prefix, kDebugGroupMain,
             strings::StrCat("after_opt_group_", grouping_name),
             VLOG_IS_ON(3) || (VLOG_IS_ON(2) &&
                               grouping == Grouping::POST_REWRITE_FOR_EXEC));

  return OkStatus();
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
