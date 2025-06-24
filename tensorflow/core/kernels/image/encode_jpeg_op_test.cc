/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using EncodeJpegWithVariableQualityTest = OpsTestBase;

TEST_F(EncodeJpegWithVariableQualityTest, FailsForInvalidQuality) {
  TF_ASSERT_OK(NodeDefBuilder("encode_op", "EncodeJpegVariableQuality")
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<uint8>(TensorShape({2, 2, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  AddInputFromArray<int32>(TensorShape({}), {200});
  absl::Status status = RunOpKernel();
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StartsWith(status.message(), "quality must be in [0,100]"));
}

}  // namespace
}  // namespace tensorflow
