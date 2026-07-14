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

// Tool that applies a series of transformations to a frozen GraphDef file.
// It takes a flexible list of transforms either on the command line, and runs
// those on the incoming graph to produce the result. This allows you to build a
// processing pipeline when preparing models for deployment.
//
// bazel build tensorflow/tools/graph_transforms/fold_constants_tool &&
// bazel-bin/tensorflow/tools/graph_transforms/fold_constants_tool \
// --in_graph=graph_def.pb \
// --out_graph=transformed_graph_def.pb \
// --inputs=input1,input2 \
// --outputs=output1,output2 \
// --transforms="fold_constants order_nodes"
//
// Parameters:
// in_graph - name of a file with a frozen GraphDef proto in binary format.
// out_graph - name of the output file to save the transformed version to.
// inputs - layer names of the nodes that will be fed data.
// outputs - layer names of the nodes that will be read from after running.
// transforms - space-separated names of the transforms to apply.
//
// List of implemented transforms:
// fold_constants - Merges constant expression subgraphs into single constants,
//   which can help reduce the number of ops and make subsequent transforms
//   optimizations more effective.
// order_nodes - Sorts the GraphDef nodes in execution order, which can help
//   simple inference engines that want to avoid complexity in their executors.

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_graph.h"

int main(int argc, char* argv[]) {
  return tensorflow::graph_transforms::ParseFlagsAndTransformGraph(argc, argv,
                                                                   true);
}
