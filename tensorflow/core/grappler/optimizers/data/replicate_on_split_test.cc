/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/replicate_on_split.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(ReplicateOnSplit, TensorSliceDataset) {
  using test::function::NDef;
  GrapplerItem item;
  Tensor tensor = test::AsTensor<int32>({32, 32});
  item.graph = test::function::GDef(
      {NDef("tensor", "Const", {}, {{"value", tensor}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeTensorSliceNode("tensor_slice_dataset", "tensor",
                                              /*replicate_on_split=*/false)});

  ReplicateOnSplit optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName("tensor_slice_dataset", output));
  int index =
      graph_utils::FindGraphNodeWithName("tensor_slice_dataset", output);
  EXPECT_TRUE(output.node(index).attr().at("replicate_on_split").b());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
