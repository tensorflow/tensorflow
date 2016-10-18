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

// Utility that transforms a model with subgraphs that evaluate to constant
// functions into the equivalent model with those subgraphs replaced by Const
// nodes. This simplifies the graph, and makes some further transformations
// easier to perform. It's often useful to run the freeze_graph tool on the
// input graph beforehand to ensure variables have been transformed to Consts.
//
// bazel-bin/tensorflow/tools/graph_transforms/fold_constants_tool \
// --in_graph=graph_def.pb \
// --out_graph=folded_graph_def.pb \
// --inputs=input1,input2 \
// --outputs=output1,output2
//
// Parameters:
// in_graph - name of a file with a frozen GraphDef proto in binary format.
// out_graph - name of the output file to save the folded version to.
// inputs - layer names of the nodes that will be fed data.
// outputs - layer names of the nodes that will be read from after running.

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"

namespace tensorflow {
namespace {

int ParseFlagsAndConvertGraph(int argc, char* argv[]) {
  string in_graph = "";
  string out_graph = "";
  string inputs_string = "";
  string outputs_string = "";
  std::vector<Flag> flag_list = {
      Flag("in_graph", &in_graph, "input graph file name"),
      Flag("out_graph", &out_graph, "output graph file name"),
      Flag("inputs", &inputs_string, "inputs"),
      Flag("outputs", &outputs_string, "outputs"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }
  if (in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty";
    return -1;
  }
  if (out_graph.empty()) {
    LOG(ERROR) << "out_graph graph can't be empty";
    return -1;
  }
  std::vector<string> inputs = str_util::Split(inputs_string, ',');
  std::vector<string> outputs = str_util::Split(outputs_string, ',');

  GraphDef graph_def;
  Status load_status = ReadBinaryProto(Env::Default(), in_graph, &graph_def);
  if (!load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << in_graph << "' failed with "
               << load_status.error_message();
    return -1;
  }

  GraphDef folded_graph_def;
  Status folding_result = graph_transforms::FoldConstants(
      graph_def, inputs, outputs, &folded_graph_def);
  if (!folding_result.ok()) {
    LOG(ERROR) << "Folding failed " << folding_result.error_message();
    return -1;
  }

  Status save_status =
      WriteBinaryProto(Env::Default(), out_graph, folded_graph_def);
  if (!save_status.ok()) {
    LOG(ERROR) << "Saving graph '" << out_graph << "' failed with "
               << save_status.error_message();
    return -1;
  }

  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::ParseFlagsAndConvertGraph(argc, argv);
}
