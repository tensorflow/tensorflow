/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/graph_utils.h"

#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::HasSubstr;
using ::testing::Pair;

TEST(GraphUtilsTest, SameGraph) {
  DatasetDef dataset1 = testing::RangeDataset(10);
  DatasetDef dataset2 = testing::RangeDataset(10);
  EXPECT_THAT(HaveEquivalentStructures(dataset1.graph(), dataset2.graph()),
              Pair(true, ""));
}

TEST(GraphUtilsTest, DifferentAttrs) {
  DatasetDef dataset1 = testing::RangeDataset(10);
  DatasetDef dataset2 = testing::RangeDataset(20);
  EXPECT_THAT(HaveEquivalentStructures(dataset1.graph(), dataset2.graph()),
              Pair(false, HasSubstr("int64_val: 20")));
}

TEST(GraphUtilsTest, DifferentStructures) {
  DatasetDef dataset1 = testing::RangeDataset(10);
  DatasetDef dataset2 = testing::RangeSquareDataset(10);
  EXPECT_THAT(HaveEquivalentStructures(dataset1.graph(), dataset2.graph()),
              Pair(false, HasSubstr("MapDataset")));
}

TEST(GraphUtilsTest, EmptyGraphs) {
  GraphDef graph1, graph2;
  EXPECT_THAT(HaveEquivalentStructures(graph1, graph2), Pair(true, ""));
}

TEST(GraphUtilsTest, OneGraphIsEmpty) {
  GraphDef graph1;
  DatasetDef dataset = testing::RangeDataset(10);
  EXPECT_THAT(HaveEquivalentStructures(graph1, dataset.graph()),
              Pair(false, HasSubstr("RangeDataset")));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
