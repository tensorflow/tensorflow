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
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void TestFakeIteratorStack() {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  TF_Operation* get_next = TF_MakeFakeIteratorGetNextWithDatasets(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  CSession csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  const float base_value = 42.0;
  for (int i = 0; i < 3; ++i) {
    csession.SetOutputs({get_next});
    csession.Run(s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_Tensor* out = csession.output_tensor(0);
    ASSERT_TRUE(out != nullptr);
    ASSERT_EQ(TF_FLOAT, TF_TensorType(out));
    ASSERT_EQ(0, TF_NumDims(out));  // scalar
    ASSERT_EQ(sizeof(float), TF_TensorByteSize(out));
    float* output_contents = static_cast<float*>(TF_TensorData(out));
    ASSERT_EQ(base_value + i, *output_contents);
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

TEST(CAPI_EXPERIMENTAL, FakeIteratorGetNext) { TestFakeIteratorStack(); }

TEST(CAPI_EXPERIMENTAL, ImagenetIteratorGetNext) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  const string file_path = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(), "c/testdata/tf_record");
  VLOG(1) << "data file path is " << file_path;
  const int batch_size = 64;
  TF_Operation* get_next = TF_MakeFileBasedIteratorGetNextWithDatasets(
      graph, file_path.c_str(), batch_size, /*is_mnist*/ false, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  CSession csession(graph, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

  // Run the graph.
  // The two output tensors should look like:
  // Tensor("IteratorGetNext:0", shape=(batch_size, 224, 224, 3), dtype=float32)
  // Tensor("IteratorGetNext:1", shape=(batch_size, ), dtype=int32)
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Running iter " << i;
    csession.SetOutputs({{get_next, 0}, {get_next, 1}});
    csession.Run(s);
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);

    {
      TF_Tensor* image = csession.output_tensor(0);
      ASSERT_TRUE(image != nullptr);
      ASSERT_EQ(TF_FLOAT, TF_TensorType(image));
      // Confirm shape is 224 X 224 X 3
      ASSERT_EQ(4, TF_NumDims(image));
      ASSERT_EQ(batch_size, TF_Dim(image, 0));
      ASSERT_EQ(224, TF_Dim(image, 1));
      ASSERT_EQ(224, TF_Dim(image, 2));
      ASSERT_EQ(3, TF_Dim(image, 3));
      ASSERT_EQ(sizeof(float) * batch_size * 224 * 224 * 3,
                TF_TensorByteSize(image));
    }

    {
      TF_Tensor* label = csession.output_tensor(1);
      ASSERT_TRUE(label != nullptr);
      ASSERT_EQ(TF_INT32, TF_TensorType(label));
      ASSERT_EQ(1, TF_NumDims(label));
      ASSERT_EQ(batch_size, TF_Dim(label, 0));
      ASSERT_EQ(sizeof(int32) * batch_size, TF_TensorByteSize(label));
    }
  }

  // Clean up
  csession.CloseAndDelete(s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(s);
}

}  // namespace
}  // namespace tensorflow
