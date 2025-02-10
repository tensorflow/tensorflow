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

NodeDef MakeBatchV2Node(absl::string_view name,
                        absl::string_view input_node_name,
                        absl::string_view batch_size_node_name,
                        absl::string_view drop_remainder_node_name,
                        bool parallel_copy) {
  return test::function::NDef(
      name, "BatchDatasetV2",
      {string(input_node_name), string(batch_size_node_name),
       string(drop_remainder_node_name)},
      {{"parallel_copy", parallel_copy},
       {"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}}});
}

NodeDef MakeParallelBatchNode(absl::string_view name,
                              absl::string_view input_node_name,
                              absl::string_view batch_size_node_name,
                              absl::string_view num_parallel_calls_node_name,
                              absl::string_view drop_remainder_node_name,
                              absl::string_view deterministic) {
  return test::function::NDef(
      name, "ParallelBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}},
       {"deterministic", string(deterministic)}});
}

NodeDef MakeCacheV2Node(absl::string_view name,
                        absl::string_view input_node_name,
                        absl::string_view filename_node_name,
                        absl::string_view cache_node_name) {
  return test::function::NDef(
      name, "CacheDatasetV2",
      {
          string(input_node_name),
          string(filename_node_name),
          string(cache_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
      });
}

NodeDef MakeFilterNode(absl::string_view name,
                       absl::string_view input_node_name,
                       absl::string_view function_name) {
  return test::function::NDef(
      name, "FilterDataset", {string(input_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}}});
}

NodeDef MakeMapAndBatchNode(absl::string_view name,
                            absl::string_view input_node_name,
                            absl::string_view batch_size_node_name,
                            absl::string_view num_parallel_calls_node_name,
                            absl::string_view drop_remainder_node_name,
                            absl::string_view function_name) {
  return test::function::NDef(
      name, "MapAndBatchDataset",
      {string(input_node_name), string(batch_size_node_name),
       string(num_parallel_calls_node_name), string(drop_remainder_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}}});
}

NodeDef MakeMapNode(absl::string_view name, absl::string_view input_node_name,
                    absl::string_view function_name) {
  return test::function::NDef(
      name, "MapDataset", {string(input_node_name)},
      {{"f", FunctionDefHelper::FunctionRef(string(function_name))},
       {"Targuments", {}},
       {"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}}});
}

NodeDef MakeParallelInterleaveV2Node(
    absl::string_view name, absl::string_view input_node_name,
    absl::string_view cycle_length_node_name,
    absl::string_view block_length_node_name,
    absl::string_view num_parallel_calls_node_name,
    absl::string_view function_name, bool sloppy) {
  return test::function::NDef(
      name, "ParallelInterleaveDatasetV2",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelInterleaveV4Node(
    absl::string_view name, absl::string_view input_node_name,
    absl::string_view cycle_length_node_name,
    absl::string_view block_length_node_name,
    absl::string_view num_parallel_calls_node_name,
    absl::string_view function_name, absl::string_view deterministic) {
  return test::function::NDef(
      name, "ParallelInterleaveDatasetV4",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeInterleaveNode(absl::string_view name,
                           absl::string_view input_node_name,
                           absl::string_view cycle_length_node_name,
                           absl::string_view block_length_node_name,
                           absl::string_view function_name,
                           absl::string_view deterministic) {
  return test::function::NDef(
      name, "InterleaveDataset",
      {string(input_node_name), string(cycle_length_node_name),
       string(block_length_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"deterministic", string(deterministic)},
      });
}

NodeDef MakeParallelMapNode(absl::string_view name,
                            absl::string_view input_node_name,
                            absl::string_view num_parallel_calls_node_name,
                            absl::string_view function_name, bool sloppy) {
  return test::function::NDef(
      name, "ParallelMapDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeParallelMapV2Node(absl::string_view name,
                              absl::string_view input_node_name,
                              absl::string_view num_parallel_calls_node_name,
                              absl::string_view function_name,
                              absl::string_view deterministic,
                              bool use_unbounded_threadpool) {
  return test::function::NDef(
      name, "ParallelMapDatasetV2",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"f", FunctionDefHelper::FunctionRef(string(function_name))},
          {"Targuments", {}},
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"deterministic", string(deterministic)},
          {"use_unbounded_threadpool", use_unbounded_threadpool},
      });
}

NodeDef MakeParseExampleNode(absl::string_view name,
                             absl::string_view input_node_name,
                             absl::string_view num_parallel_calls_node_name,
                             bool sloppy) {
  return test::function::NDef(
      name, "ParseExampleDataset",
      {string(input_node_name), string(num_parallel_calls_node_name)},
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"sloppy", sloppy},
      });
}

NodeDef MakeShuffleV2Node(absl::string_view name,
                          absl::string_view input_node_name,
                          absl::string_view buffer_size_node_name,
                          absl::string_view seed_generator_node_name) {
  return test::function::NDef(
      name, "ShuffleDatasetV2",
      {
          string(input_node_name),
          string(buffer_size_node_name),
          string(seed_generator_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
      });
}

NodeDef MakeTakeNode(absl::string_view name, absl::string_view input_node_name,
                     absl::string_view count_node_name) {
  return test::function::NDef(
      name, "TakeDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
      });
}

NodeDef MakeTensorSliceNode(absl::string_view name,
                            absl::string_view tensor_node_name,
                            bool replicate_on_split) {
  return test::function::NDef(
      name, "TensorSliceDataset",
      {
          string(tensor_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
          {"replicate_on_split", replicate_on_split},
      });
}

NodeDef MakeSkipNode(absl::string_view name, absl::string_view input_node_name,
                     absl::string_view count_node_name) {
  return test::function::NDef(
      name, "SkipDataset",
      {
          string(input_node_name),
          string(count_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
      });
}

NodeDef MakeShardNode(absl::string_view name, absl::string_view input_node_name,
                      absl::string_view num_shards_node_name,
                      absl::string_view index_node_name) {
  return test::function::NDef(
      name, "ShardDataset",
      {
          string(input_node_name),
          string(num_shards_node_name),
          string(index_node_name),
      },
      {
          {"output_shapes", absl::Span<const TensorShape>{}},
          {"output_types", absl::Span<const DataType>{}},
      });
}

NodeDef MakePrefetchNode(absl::string_view name,
                         absl::string_view input_node_name,
                         absl::string_view buffer_size) {
  return test::function::NDef(
      name, "PrefetchDataset", {string(input_node_name), string(buffer_size)},
      {{"output_shapes", absl::Span<const TensorShape>{}},
       {"output_types", absl::Span<const DataType>{}},
       {"slack_period", 0},
       {"legacy_autotune", true},
       {"buffer_size_min", 0}});
}

}  // namespace graph_tests_utils
}  // namespace grappler
}  // namespace tensorflow
