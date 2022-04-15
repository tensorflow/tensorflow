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

#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace grappler {
namespace graph_tests_utils {

NodeDef MakeBatchV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece batch_size_node_name,
                        StringPiece drop_remainder_node_name,
                        bool parallel_copy) {
  return test::function::NDef(
      name, "BatchDatasetV2",
      {string(input_node_name), string(batch_size_node_name),
       string(drop_remainder_node_name)},
      {{"parallel_copy", parallel_copy},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeParallelBatchNode(StringPiece name, StringPiece input_node_name,
                              StringPiece batch_size_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece drop_remainder_node_name,
                              StringPiece deterministic) {
  return test::function::NDef(
      name, "ParallelBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}},
       {"deterministic", string(deterministic)}});
}

NodeDef MakeCacheV2Node(StringPiece name, StringPiece input_node_name,
                        StringPiece filename_node_name,
                        StringPiece cache_node_name) {
  return test::function::NDef(
      name, "CacheDatasetV2",
      {
          string(input_node_name),
          string(filename_node_name),
          string(cache_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeFilterNode(StringPiece name, StringPiece input_node_name,
                       StringPiece function_name) {
  return test::function::NDef(
      name, "FilterDataset", {string(input_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeMapAndBatchNode(StringPiece name, StringPiece input_node_name,
                            StringPiece batch_size_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece drop_remainder_node_name,
                            StringPiece function_name) {
  return test::function::NDef(
      name, "MapAndBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeMapNode(StringPiece name, StringPiece input_node_name,
                    StringPiece function_name) {
  return test::function::NDef(
      name, "MapDataset", {string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}}});
}

NodeDef MakeParallelInterleaveV2Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name, bool sloppy) {
  return test::function::NDef(
      name, "ParallelInterleaveDatasetV2",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelInterleaveV4Node(StringPiece name,
                                     StringPiece input_node_name,
                                     StringPiece cycle_length_node_name,
                                     StringPiece block_length_node_name,
                                     StringPiece num_parallel_calls_node_name,
                                     StringPiece function_name,
                                     StringPiece deterministic) {
  return test::function::NDef(
      name, "ParallelInterleaveDatasetV4",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeParallelMapNode(StringPiece name, StringPiece input_node_name,
                            StringPiece num_parallel_calls_node_name,
                            StringPiece function_name, bool sloppy) {
  return test::function::NDef(
      name, "ParallelMapDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelMapV2Node(StringPiece name, StringPiece input_node_name,
                              StringPiece num_parallel_calls_node_name,
                              StringPiece function_name,
                              StringPiece deterministic) {
  return test::function::NDef(
      name, "ParallelMapDatasetV2",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeParseExampleNode(StringPiece name, StringPiece input_node_name,
                             StringPiece num_parallel_calls_node_name,
                             bool sloppy) {
  return test::function::NDef(
      name, "ParseExampleDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeShuffleV2Node(StringPiece name, StringPiece input_node_name,
                          StringPiece buffer_size_node_name,
                          StringPiece seed_generator_node_name) {
  return test::function::NDef(
      name, "ShuffleDatasetV2",
      {
          string(input_node_name),
          string(buffer_size_node_name),
          string(seed_generator_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeTakeNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name) {
  return test::function::NDef(
      name, "TakeDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeSkipNode(StringPiece name, StringPiece input_node_name,
                     StringPiece count_node_name) {
  return test::function::NDef(
      name, "SkipDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakeShardNode(StringPiece name, StringPiece input_node_name,
                      StringPiece num_shards_node_name,
                      StringPiece index_node_name) {
  return test::function::NDef(
      name, "ShardDataset",
      {
          string(input_node_name),
          string(num_shards_node_name),
          string(index_node_name),
      },
      {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}},
      });
}

NodeDef MakePrefetchNode(StringPiece name, StringPiece input_node_name,
                         StringPiece buffer_size) {
  return test::function::NDef(
      name, "PrefetchDataset", {string(input_node_name), string(buffer_size)},
      {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
       {"output_types", gtl::ArraySlice<DataType>{}},
       {"slack_period", 0},
       {"legacy_autotune", true},
       {"buffer_size_min", 0}});
}

}  // namespace graph_tests_utils
}  // namespace grappler
}  // namespace tensorflow
