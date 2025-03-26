/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/remove_compression_map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/status_matchers.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::testing::HasSubstr;

TEST(RemoveCompressionMap, Success) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("Const/_0",            // name
            "Const",               // op
            {},                    // inputs
            {{"dtype", DT_INT64},  // attrs
             {"value", 0}}),       // attrs

       NDef("Const/_1",            // name
            "Const",               // op
            {},                    // inputs
            {{"dtype", DT_INT64},  // attrs
             {"value", 10}}),      // attrs

       NDef("Const/_2",            // name
            "Const",               // op
            {},                    // inputs
            {{"dtype", DT_INT64},  // attrs
             {"value", 1}}),       // attrs

       NDef("RangeDataset/_3",  // name
            "RangeDataset",     // op
            {"Const/_0",        // inputs
             "Const/_1",        // inputs
             "Const/_2"},       // inputs
            {}),                // attrs

       NDef("Const/_4",            // name
            "Const",               // op
            {},                    // inputs
            {{"dtype", DT_INT64},  // attrs
             {"value", -1}}),      // attrs

       graph_tests_utils::MakeParallelMapV2Node(
           /*name=*/"ParallelMapDatasetV2/_5",
           /*input_node_name=*/"RangeDataset/_3",
           /*num_parallel_calls_node_name=*/"Const/_4",
           /*function_name=*/"__inference_Dataset_map_lambda_10",
           /*deterministic=*/"default",
           /*use_unbounded_threadpool=*/false),

       NDef("dataset",                    // name
            "_Retval",                    // op
            {"ParallelMapDatasetV2/_5"},  // inputs
            {{"T", DT_VARIANT}}),         // attrs

       NDef("Sink",                       // name
            "Identity",                   // op
            {"ParallelMapDatasetV2/_5"},  // inputs
            {{"T", DT_VARIANT}})},        // attrs

      {FunctionDefHelper::Create(
          "__inference_Dataset_map_lambda_10",  // function_name
          {"args_0: int64"},                    // in_def
          {"identity: variant"},                // out_def
          {},                                   // attr_def
          {
              // node_def
              {{"CompressElement"},           // name
               "CompressElement",             // op
               {"args_0"},                    // input
               {{"input_types", DT_INT64}}},  // attrs

              {{"Identity"},                      // name
               "Identity",                        // op
               {"CompressElement:compressed:0"},  // input
               {{"T", DT_VARIANT}}},              // attrs
          },
          {})});  // ret_def

  RemoveCompressionMap optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("dataset", output);
  EXPECT_EQ(output.node(index).input(0), "RangeDataset/_3");
}

TEST(RemoveCompressionMap, FailureNoMap) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef({NDef("Const/_0",            // name
                                          "Const",               // op
                                          {},                    // inputs
                                          {{"dtype", DT_INT64},  // attrs
                                           {"value", 0}}),       // attrs

                                     NDef("Const/_1",            // name
                                          "Const",               // op
                                          {},                    // inputs
                                          {{"dtype", DT_INT64},  // attrs
                                           {"value", 10}}),      // attrs

                                     NDef("Const/_2",            // name
                                          "Const",               // op
                                          {},                    // inputs
                                          {{"dtype", DT_INT64},  // attrs
                                           {"value", 1}}),       // attrs

                                     NDef("RangeDataset/_3",  // name
                                          "RangeDataset",     // op
                                          {"Const/_0",        // inputs
                                           "Const/_1",        // inputs
                                           "Const/_2"},       // inputs
                                          {}),                // attrs

                                     NDef("dataset",             // name
                                          "_Retval",             // op
                                          {"RangeDataset/_3"},   // inputs
                                          {{"T", DT_VARIANT}}),  // attrs

                                     NDef("Sink",                  // name
                                          "Identity",              // op
                                          {"RangeDataset/_3"},     // inputs
                                          {{"T", DT_VARIANT}})});  // attrs

  RemoveCompressionMap optimizer;
  GraphDef output;
  ASSERT_THAT(optimizer.Optimize(nullptr, item, &output),
              testing::StatusIs(error::INTERNAL,
                                HasSubstr("Compression function not found.")));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
