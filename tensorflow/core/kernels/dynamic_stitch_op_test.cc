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

#include <functional>
#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class DynamicStitchOpTest : public OpsTestBase {
 protected:
  void MakeOp(int n, DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "DynamicStitch")
                     .Input(FakeInput(n, DT_INT32))
                     .Input(FakeInput(n, dt))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(DynamicStitchOpTest, Simple_OneD) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({5}), {10, 60, 20, 30, 50});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {0, 10, 20, 30, 40, 50, 60, 70});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(DynamicStitchOpTest, Simple_TwoD) {
  MakeOp(3, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({2}), {1, 6});
  AddInputFromArray<int32>(TensorShape({3}), {2, 3, 5});
  AddInputFromArray<float>(TensorShape({3, 2}), {0, 1, 40, 41, 70, 71});
  AddInputFromArray<float>(TensorShape({2, 2}), {10, 11, 60, 61});
  AddInputFromArray<float>(TensorShape({3, 2}), {20, 21, 30, 31, 50, 51});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8, 2}));
  test::FillValues<float>(&expected, {0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50,
                                      51, 60, 61, 70, 71});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(DynamicStitchOpTest, Error_IndicesMultiDimensional) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({1, 5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({5}), {10, 60, 20, 30, 50});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("data[1].shape = [5] does not start with "
                            "indices[1].shape = [1,5]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataNumDimsMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({1, 5}), {10, 60, 20, 30, 50});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("data[1].shape = [1,5] does not start with "
                            "indices[1].shape = [5]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataDimSizeMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 5});
  AddInputFromArray<int32>(TensorShape({4}), {1, 6, 2, 3});
  AddInputFromArray<float>(TensorShape({3, 1}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({4, 2}),
                           {10, 11, 60, 61, 20, 21, 30, 31});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Need data[0].shape[1:] = data[1].shape[1:], "
                            "got data[0].shape = [3,1], data[1].shape = [4,2]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataAndIndicesSizeMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({4}), {10, 60, 20, 30});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString())
          .contains(
              "data[1].shape = [4] does not start with indices[1].shape = [5]"))
      << s;
}

}  // namespace
}  // namespace tensorflow
