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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/print_model_analysis.h"

#include <stdio.h>
#include <memory>
#include <utility>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_stats.h"

namespace tensorflow {
namespace tfprof {
string PrintModelAnalysis(const string* graph, const string* run_meta,
                          const string* op_log, const string* command,
                          const Options* options) {
  CHECK(graph) << "graph mustn't be null";
  CHECK(command) << "command mustn't be null";
  CHECK(options) << "options mustn't be null";
  std::unique_ptr<GraphDef> graph_ptr(new GraphDef());
  graph_ptr->ParseFromString(*graph);

  std::unique_ptr<RunMetadata> run_meta_ptr;
  if (run_meta) {
    run_meta_ptr.reset(new RunMetadata());
    run_meta_ptr->ParseFromString(*run_meta);
  }

  std::unique_ptr<OpLog> op_log_ptr;
  if (op_log) {
    op_log_ptr.reset(new OpLog());
    op_log_ptr->ParseFromString(*op_log);
  }

  std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader;

  TFStats tf_stats(std::move(graph_ptr), std::move(run_meta_ptr),
                   std::move(op_log_ptr), std::move(ckpt_reader));

  if (options->dump_to_file.empty()) {
    printf("\n=========================Options=============================\n");
    printf("%s", options->ToString().c_str());
    printf("\n==================Model Analysis Report======================\n");
    TFProfNode root(tf_stats.PrintGraph(*command, *options));
    printf("\n======================End of Report==========================\n");
    fflush(stdout);
    return root.SerializeAsString();
  }
  return tf_stats.PrintGraph(*command, *options).SerializeAsString();
}
}  // namespace tfprof
}  // namespace tensorflow
