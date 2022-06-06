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

// Creates a test NodeDef for BatchDatasetV2.
NodeDef MakeBatchV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece batch_size_node_name,
                        StringPiece drop_remainder_node_name,
                        bool parallel_copy);

// Creates a test NodeDef for ParallelBatchDataset.
NodeDef MakeParallelBatchNode(StringPiece name, StringPiece input_node_name,
                              StringPiece batch_size_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece drop_remainder_node_name,
                              StringPiece deterministic);

// Creates a test NodeDef for ShuffleDatasetV2.
NodeDef MakeCacheV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece filename_node_name,
                        StringPiece cache_node_name);

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

// Creates a test NodeDef for ParallelInterleaveDatasetV2.
NodeDef MakeParallelInterleaveV2Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name, bool sloppy);

// Creates a test NodeDef for ParallelInterleaveDatasetV4.
NodeDef MakeParallelInterleaveV4Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name,
                                     StringPiece deterministic);

// Creates a test NodeDef for ParallelMapDataset.
NodeDef MakeParallelMapNode(StringPiece name, StringPiece input_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece function_name, bool sloppy);

// Creates a test NodeDef for ParallelMapDatasetV2.
NodeDef MakeParallelMapV2Node(StringPiece name, StringPiece input_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece function_name,
                              StringPiece deterministic);

// Creates a test NodeDef for ParseExampleDataset.
NodeDef MakeParseExampleNode(StringPiece name, StringPiece input_node_name,
                             StringPiece num_parallel_calls_node_name,
                             bool sloppy);

// Creates a test NodeDef for ShuffleDatasetV2.
NodeDef MakeShuffleV2Node(StringPiece name, StringPiece input_node_name,
                          StringPiece buffer_size_node_name,
                          StringPiece seed_generator_node_name);

// Creates a test NodeDef for TakeDataset.
NodeDef MakeTakeNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name);

// Creates a test NodeDef for TensorSliceDataset.
NodeDef MakeTensorSliceNode(StringPiece name, StringPiece tensor_node_name,
                            bool replicate_on_split);

// Creates a test NodeDef for SkipDataset.
NodeDef MakeSkipNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name);

// Creates a test NodeDef for ShardDataset.
NodeDef MakeShardNode(StringPiece name, StringPiece input_node_name,
                      StringPiece num_shards_node_name,
                      StringPiece index_node_name);

// Creates a test NodeDef for PrefetchDataset.
NodeDef MakePrefetchNode(StringPiece name, StringPiece input_node_name,
                         StringPiece buffer_size);

}  // namespace graph_tests_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_GRAPH_TEST_UTILS_H_
