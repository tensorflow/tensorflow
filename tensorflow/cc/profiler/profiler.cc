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
#include "tensorflow/cc/profiler/profiler.h"

namespace tensorflow {
namespace tfprof {

Profiler::Profiler(const GraphDef& graph) {
  std::unique_ptr<GraphDef> graph_ptr(new GraphDef());
  *graph_ptr = graph;
  stats_.reset(new TFStats(std::move(graph_ptr), nullptr, nullptr, nullptr));
}

void Profiler::AddStep(int64 step, const RunMetadata& run_meta) {
  std::unique_ptr<RunMetadata> run_meta_ptr(new RunMetadata());
  *run_meta_ptr = run_meta;
  stats_->AddRunMeta(step, std::move(run_meta_ptr));
}

GraphNodeProto Profiler::ProfileGraph(const Options& options) {
  stats_->BuildView(kCmds[1]);
  return stats_->ShowGraphNode(kCmds[1], options);
}

GraphNodeProto Profiler::ProfileNameScope(const Options& options) {
  stats_->BuildView(kCmds[0]);
  return stats_->ShowGraphNode(kCmds[0], options);
}

MultiGraphNodeProto Profiler::ProfileOperations(const Options& options) {
  stats_->BuildView(kCmds[3]);
  return stats_->ShowMultiGraphNode(kCmds[3], options);
}

Status Profiler::SerializeToString(string* content) {
  if (!content) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Cannot use null string pointer for SerializeToString.");
  }
  stats_->SerializeToString(content);
  return Status::OK();
}

}  // namespace tfprof
}  // namespace tensorflow
