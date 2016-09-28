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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_

#include <map>
#include <memory>
#include <set>
#include <string>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_graph.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_node.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_scope.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_show.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_output.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {

class TFStats {
 public:
  TFStats(std::unique_ptr<GraphDef> graph,
          std::unique_ptr<RunMetadata> run_meta, std::unique_ptr<OpLog> op_log,
          std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader);
  ~TFStats() {}

  // Prints the results to stdout. Also returns the printed output in
  // a proto.
  const TFProfNode& PrintGraph(const string& cmd, const Options& opts);

 private:
  void ParseGraph();

  void ParseOpLog();

  void ParseRunMeta();

  std::unique_ptr<TFScope> scope_view_;
  std::unique_ptr<TFGraph> graph_view_;
  std::unique_ptr<GraphDef> graph_;
  std::unique_ptr<RunMetadata> run_meta_;
  std::unique_ptr<OpLog> op_log_;
  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader_;
  // Store TFNode instead of TFNode* to avoid large number of dynamic alloc.
  std::map<string, TFNode> nodes_map_;
  TFProfNode empty_node_;
};

}  // namespace tfprof
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TFPROF_TOOLS_TFPROF_INTERNAL_TFPROF_STATS_H_
