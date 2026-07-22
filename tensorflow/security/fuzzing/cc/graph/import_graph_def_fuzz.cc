/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"

// Raw-bytes fuzzer for tensorflow::ImportGraphDef -- the C++ entry point
// exercised by Python's tf.import_graph_def().
//
// Why this exists alongside the structured-proto
// //tensorflow/core/common_runtime:graph_constructor_fuzz (which takes
// a typed GraphDef argument):
//
//  * This one takes std::string_view, parses the bytes as a GraphDef
//    wire-format protobuf, and invokes ImportGraphDef. The structured
//    fuzzer cannot synthesize certain specific graph shapes reliably --
//    e.g. FunctionDefLibrary entries where two functions reference each
//    other by name, which is the shape needed to trigger the
//    import-graph-def mutual-recursion stack overflow.
//  * Raw-bytes input also allows straightforward reproduction of a
//    specific crashing .pb from disk via the standard libFuzzer repro
//    workflow (`./binary <testcase>`).

namespace {

void FuzzImportGraphDef(std::string_view data) {
  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromArray(data.data(),
                                static_cast<int>(data.size()))) {
    return;
  }

  tensorflow::Graph graph(tensorflow::OpRegistry::Global());
  tensorflow::ImportGraphDefOptions opts;
  // Ignore the returned status -- we only care about hard crashes
  // (SIGSEGV, ASan reports, stack overflow), not well-formed errors.
  (void)tensorflow::ImportGraphDef(opts, graph_def, &graph,
                                   /*refiner=*/nullptr);
}
FUZZ_TEST(CC_FUZZING, FuzzImportGraphDef);

}  // namespace
