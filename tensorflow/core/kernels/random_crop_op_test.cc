/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include <gtest/gtest.h>
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

class RandomCropOpTest : public OpsTestBase {
 protected:
  RandomCropOpTest() {
    RequireDefaultOps();
    EXPECT_OK(NodeDefBuilder("random_crop_op", "RandomCrop")
                  .Input(FakeInput(DT_UINT8))
                  .Input(FakeInput(DT_INT64))
                  .Attr("T", DT_UINT8)
                  .Finalize(node_def()));
    EXPECT_OK(InitOp());
  }
};

TEST_F(RandomCropOpTest, Basic) {
  AddInputFromArray<uint8>(TensorShape({1, 2, 1}), {2, 2});
  AddInputFromArray<int64>(TensorShape({2}), {1, 1});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_UINT8, TensorShape({1, 1, 1}));
  test::FillValues<uint8>(&expected, {2});
  test::ExpectTensorEqual<uint8>(expected, *GetOutput(0));
}

TEST_F(RandomCropOpTest, SameSizeOneChannel) {
  AddInputFromArray<uint8>(TensorShape({2, 1, 1}), {1, 2});
  AddInputFromArray<int64>(TensorShape({2}), {2, 1});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_UINT8, TensorShape({2, 1, 1}));
  test::FillValues<uint8>(&expected, {1, 2});
  test::ExpectTensorEqual<uint8>(expected, *GetOutput(0));
}

TEST_F(RandomCropOpTest, SameSizeMultiChannel) {
  AddInputFromArray<uint8>(TensorShape({2, 1, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int64>(TensorShape({2}), {2, 1});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_UINT8, TensorShape({2, 1, 3}));
  test::FillValues<uint8>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<uint8>(expected, *GetOutput(0));
}

}  // namespace tensorflow
