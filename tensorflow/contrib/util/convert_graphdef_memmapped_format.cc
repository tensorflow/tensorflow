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

// Utility that converts a "frozen" inference graph (output from the
// freeze_graph utility) into a format in which large Const ops are converted to
// ImmutableConst ops which are memmapped when the graph is executed by
// TensorFlow.
//
//  tensorflow/contrib/util/convert_graphdef_memmapped_format
//        --in_graph=frozen.model --out_graph=memmapped.mmodel
//
// Parameters:
// in_graph - name of a file with a frozen GraphDef proto in binary format
// out_graph - name of the output file, where the graph in memmapped format will
// be saved.
// min_conversion_size_bytes - tensors with fewer than this many bytes of data
// will not be converted to ImmutableConst format, and kept in the graph.

#include "tensorflow/contrib/util/convert_graphdef_memmapped_format_lib.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

int ParseFlagsAndConvertGraph(int argc, char* argv[]) {
  string in_graph = "";
  string out_graph = "";
  int min_conversion_tensor_size = 10000;
  const bool parse_result = ParseFlags(
      &argc, argv,
      {// input graph
       Flag("in_graph", &in_graph),
       // output graph
       Flag("out_graph", &out_graph),
       // constants with tensors that have less than this number elements won't
       // be converted into ImmutableConst (be memmapped).
       Flag("min_conversion_tensor_size", &min_conversion_tensor_size)});
  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);
  if (!parse_result) {
    LOG(ERROR) << "Error parsing command-line flags.";
    return -1;
  }
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1];
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
  if (min_conversion_tensor_size <= 0) {
    LOG(ERROR) << "min_conversion_tensor_size must be > 0";
    return -1;
  }
  const auto result = ConvertConstantsToImmutable(in_graph, out_graph,
                                                  min_conversion_tensor_size);
  if (!result.ok()) {
    LOG(ERROR) << "Conversion failed " << result.error_message();
    return -1;
  }
  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) {
  return tensorflow::ParseFlagsAndConvertGraph(argc, argv);
}
