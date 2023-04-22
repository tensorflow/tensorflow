/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {

class StringFormatGraphTest : public OpsTestBase {
 protected:
  Status Init(int num_inputs, DataType input_type,
              const string& template_ = "%s", const string& placeholder = "%s",
              int summarize = 3) {
    TF_CHECK_OK(NodeDefBuilder("op", "StringFormat")
                    .Input(FakeInput(num_inputs, input_type))
                    .Attr("template", template_)
                    .Attr("placeholder", placeholder)
                    .Attr("summarize", summarize)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(StringFormatGraphTest, Int32Success_7) {
  TF_ASSERT_OK(Init(1, DT_INT32, "First tensor: %s"));

  AddInputFromArray<int32>(TensorShape({7}), {1, 2, 3, 4, 5, 6, 7});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({}));
  test::FillValues<tstring>(&expected, {"First tensor: [1 2 3 ... 5 6 7]"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(StringFormatGraphTest, Int32Success_3_3) {
  TF_ASSERT_OK(Init(1, DT_INT32, "First tensor: %s", "%s", 1));

  AddInputFromArray<int32>(TensorShape({3, 3}), {1, 2, 3, 4, 5, 6, 7, 8, 9});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({}));
  test::FillValues<tstring>(&expected, {"First tensor: [[1 ... 3]\n ..."
                                        "\n [7 ... 9]]"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

}  // end namespace
}  // end namespace tensorflow
