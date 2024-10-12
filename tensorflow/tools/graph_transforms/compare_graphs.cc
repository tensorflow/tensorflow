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

// Compares two TensorFlow graphs to see if their meaning is the same. This is a
// semantic comparison that's intended to show whether the graphs should produce
// the same results, and so ignores details like version numbers or node
// ordering that don't affect the output. To use it, run something like this:
//
// bazel build tensorflow/tools/graph_transforms:compare_graphs
// bazel-bin/tensorflow/tools/graph_transforms/compare_graphs a.pb b.pb
//
// The return value is 0 if the graphs are equal, 1 if they're different, and -1
// if there was a problem.

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/equal_graph_def.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {
namespace {

int ParseFlagsAndCompareGraphs(int argc, char* argv[]) {
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);

  if (argc != 3) {
    LOG(ERROR) << "compare_graphs expects two file names as arguments";
    return -1;
  }

  GraphDef a;
  absl::Status a_load_status = LoadTextOrBinaryGraphFile(argv[1], &a);
  if (!a_load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << argv[1] << "' failed with "
               << a_load_status.message();
    return -1;
  }

  GraphDef b;
  absl::Status b_load_status = LoadTextOrBinaryGraphFile(argv[2], &b);
  if (!b_load_status.ok()) {
    LOG(ERROR) << "Loading graph '" << argv[2] << "' failed with "
               << b_load_status.message();
    return -1;
  }

  string diff;
  if (EqualGraphDef(a, b, &diff)) {
    std::cout << "Graphs are equal." << std::endl;
    return 0;
  } else {
    std::cout << diff << std::endl;
    return 1;
  }
}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::graph_transforms::ParseFlagsAndCompareGraphs(argc, argv);
}
