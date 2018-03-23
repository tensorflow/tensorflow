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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/c_test_util.h"

namespace tensorflow {
namespace {

void TestIteratorStack() {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  TF_Function* dataset_func = nullptr;

  TF_Operation* get_next =
      TF_MakeIteratorGetNextWithDatasets(graph, "dummy_path", &dataset_func, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  ASSERT_NE(dataset_func, nullptr);
  TF_DeleteFunction(dataset_func);

  CSession csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  for (int i = 0; i < 1; ++i) {
    csession.SetOutputs({get_next});
    csession.Run(s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_Tensor* out = csession.output_tensor(0);
    ASSERT_TRUE(out != nullptr);
    EXPECT_EQ(TF_INT32, TF_TensorType(out));
    EXPECT_EQ(0, TF_NumDims(out));  // scalar
    ASSERT_EQ(sizeof(int32), TF_TensorByteSize(out));
    int32* output_contents = static_cast<int32*>(TF_TensorData(out));
    EXPECT_EQ(1, *output_contents);
  }

  // This should error out since we've exhausted the iterator.
  csession.Run(s);
  ASSERT_EQ(TF_OUT_OF_RANGE, TF_GetCode(s)) << TF_Message(s);

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

TEST(CAPI_EXPERIMENTAL, IteratorGetNext) { TestIteratorStack(); }

}  // namespace
}  // namespace tensorflow
