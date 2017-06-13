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

#ifndef THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_
#define THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_

#include <map>
#include <memory>
#include <set>
#include <string>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tools/tfprof/internal/tfprof_code.h"
#include "tensorflow/tools/tfprof/internal/tfprof_graph.h"
#include "tensorflow/tools/tfprof/internal/tfprof_node.h"
#include "tensorflow/tools/tfprof/internal/tfprof_op.h"
#include "tensorflow/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/tools/tfprof/internal/tfprof_scope.h"
#include "tensorflow/tools/tfprof/internal/tfprof_show.h"
#include "tensorflow/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/tools/tfprof/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {

class TFStats {
 public:
  TFStats(std::unique_ptr<GraphDef> graph,
          std::unique_ptr<RunMetadata> run_meta, std::unique_ptr<OpLog> op_log,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);
  ~TFStats() {}

  const std::map<string, std::unique_ptr<TFGraphNode>>& nodes() const {
    return nodes_map_;
  }

  // Organize the TensorFlow model as different types of views, and generate
  // outputs for profiling.
  const TFGraphNodeProto& ShowGraphNode(const string& cmd, const Options& opts);
  const TFMultiGraphNodeProto& ShowMultiGraphNode(const string& cmd,
                                                  const Options& opts);

  // Add a step of run time meta data.
  void ParseRunMeta(int64 step, std::unique_ptr<RunMetadata> run_meta);
  // Add tfprof operation meta data, such as customized op type, float_ops,
  // and code traces.
  void ParseOpLog(std::unique_ptr<OpLog> op_log);

  // For test purpose only.
  void AddNodeForTest(const string& name, std::unique_ptr<TFGraphNode> node);

 private:
  bool Validate(const Options& opts);

  void ParseGraph();

  std::set<int64> steps_;
  std::unique_ptr<GraphDef> graph_;
  std::unique_ptr<TFScope> scope_view_;
  std::unique_ptr<TFGraph> graph_view_;
  std::unique_ptr<TFCode> code_view_;
  std::unique_ptr<TFOp> op_view_;
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader_;
  // Store TFGraphNode instead of TFGraphNode* to avoid large number of
  // dynamic alloc.
  std::map<string, std::unique_ptr<TFGraphNode>> nodes_map_;
  TFGraphNodeProto empty_graph_node_;
  TFMultiGraphNodeProto empty_multi_graph_node_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_
