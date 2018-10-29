/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace grappler {
namespace graph_tests_utils {

// Creates a test NodeDef for FilterDataset.
NodeDef MakeFilterNode(StringPiece name, StringPiece input_node_name,
                       StringPiece function_name = "IsZero");

// Creates a test NodeDef for MapDataset.
NodeDef MakeMapNode(StringPiece name, StringPiece input_node_name,
                    StringPiece function_name = "XTimesTwo");

// Creates a test NodeDef for MapAndBatchDataset.
NodeDef MakeMapAndBatchNode(StringPiece name, StringPiece input_node_name,
                            StringPiece batch_size_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece drop_remainder_node_name,
                            StringPiece function_name = "XTimesTwo");

// Creates a test NodeDef for ParallelInterleaveDataset.
NodeDef MakeParallelInterleaveNode(StringPiece name,
                                   StringPiece input_node_name,
                                   StringPiece cycle_length_node_name,
                                   StringPiece block_length_node_name,
                                   StringPiece num_parallel_calls_node_name,
                                   StringPiece function_name, bool sloppy);

// Creates a test NodeDef for ParallelMapDataset.
NodeDef MakeParallelMapNode(StringPiece name, StringPiece input_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece function_name, bool sloppy);

// Creates a test NodeDef for ParseExampleDataset.
NodeDef MakeParseExampleNode(StringPiece name, StringPiece input_node_name,
                             StringPiece num_parallel_calls_node_name,
                             bool sloppy);

}  // end namespace graph_tests_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_
