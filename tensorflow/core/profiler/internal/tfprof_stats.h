/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// Core API of tfprof.
// 1. Load protos generated from a tensorflow model.
// 2. Build in-memory representations of the tensorflow model, annotate the
//    representation with various stats, such as params,times,memory,etc.
// 3. Accept command and options to selectively aggregate stats for analysis
//    and print out the results.

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_

#include <map>
#include <memory>
#include <set>
#include <string>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/internal/tfprof_code.h"
#include "tensorflow/core/profiler/internal/tfprof_graph.h"
#include "tensorflow/core/profiler/internal/tfprof_node.h"
#include "tensorflow/core/profiler/internal/tfprof_op.h"
#include "tensorflow/core/profiler/internal/tfprof_scope.h"
#include "tensorflow/core/profiler/internal/tfprof_show.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

class TFStats {
 public:
  TFStats(std::unique_ptr<GraphDef> graph,
          std::unique_ptr<RunMetadata> run_meta,
          std::unique_ptr<OpLogProto> op_log,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);

  TFStats(const string& filename,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);

  ~TFStats() {}

  const std::map<string, std::unique_ptr<TFGraphNode>>& nodes() const {
    return nodes_map_;
  }
  const std::set<int64_t>& steps() const { return steps_; }
  bool has_code_traces() const { return has_code_traces_; }
  double run_coverage() const {
    return covered_nodes_.size() / (nodes_map_.size() + 1e-10);
  }

  void BuildView(const string& cmd);
  void BuildAllViews();

  // Note: Must first BuildView(view_foo) before ShowXXX(view_foo) methods.
  //
  // Organize the TensorFlow model as different types of views, and generate
  // outputs for profiling.
  // TODO(xpan): Should it return reference here?
  const GraphNodeProto& ShowGraphNode(const string& cmd,
                                      const Options& opts) const;
  const MultiGraphNodeProto& ShowMultiGraphNode(const string& cmd,
                                                const Options& opts) const;

  // Add a (partial) graph to existing graph.
  void AddGraph(std::unique_ptr<GraphDef> graph);

  // Add a step of run time meta data.
  void AddRunMeta(int64_t step, std::unique_ptr<RunMetadata> run_meta);
  // Add tfprof operation meta data, such as customized op type, float_ops,
  // and code traces.
  void AddOpLogProto(std::unique_ptr<OpLogProto> op_log);

  void SerializeToString(string* content);
  void WriteProfile(const string& filename);

  // For test purpose only.
  void AddNodeForTest(int64_t step, std::unique_ptr<TFGraphNode> node);

 private:
  bool Validate(const Options& opts) const;
  string MaybeReportMissingTrace() const;

  std::set<int64_t> steps_;
  bool has_code_traces_;
  bool miss_accelerator_stream_;
  std::unique_ptr<TFScope> scope_view_;
  std::unique_ptr<TFGraph> graph_view_;
  std::unique_ptr<TFCode> code_view_;
  std::unique_ptr<TFOp> op_view_;
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader_;
  // TODO(xpan): Store TFGraphNode instead of TFGraphNode* to avoid large
  // number of dynamic alloc.
  // Maps from graph node name to TFGraphNode.
  std::map<string, std::unique_ptr<TFGraphNode>> nodes_map_;
  GraphNodeProto empty_graph_node_;
  MultiGraphNodeProto empty_multi_graph_node_;

  std::map<int64_t, string> id_to_string_;
  // Graph nodes covered by RunMetadata, that is traced with run time stats.
  std::set<int64_t> covered_nodes_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_TFPROF_STATS_H_
