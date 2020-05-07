/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/ops_testutil.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST_F(OpsTestBase, ScopedStepContainer) {
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {""});
  TF_EXPECT_OK(RunOpKernel());
  EXPECT_TRUE(step_container_ != nullptr);
}

// Verify that a Resource input can be added to the test kernel.
TEST_F(OpsTestBase, ResourceVariableInput) {
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  Var* var = new Var(DT_STRING);
  AddResourceInput("" /* container */, "Test" /* name */, var);
  TF_ASSERT_OK(RunOpKernel());
  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->dtype(), DT_RESOURCE);
}

}  // namespace tensorflow
