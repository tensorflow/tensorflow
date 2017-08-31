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
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests for the switch op
class SwitchOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("op", "Switch")
                     .Input(FakeInput(dt))
                     .Input(FakeInput())
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SwitchOpTest, Int32Success_6_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, Int32Success_6_s1) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

TEST_F(SwitchOpTest, Int32Success_2_3_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, StringSuccess_s1) {
  Initialize(DT_STRING);
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  AddInputFromArray<bool>(TensorShape({}), {true});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

class AbortOpTest : public OpsTestBase {
 protected:
};

#ifdef PLATFORM_WINDOWS
#define SIGABRT 3

class KilledBySignal {
 public:
  explicit KilledBySignal(int signum) : signum_(signum) {}
  bool operator()(int exit_status) const { return exit_status == signum_; }
 private:
  const int signum_;
};
#else
#define KilledBySignal ::testing::KilledBySignal
#endif

// Pass an error message to the op.
TEST_F(AbortOpTest, pass_error_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("error_msg", "abort_op_test")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; abort_op_test");
}

// Use the default error message.
TEST_F(AbortOpTest, default_msg) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort").Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), KilledBySignal(SIGABRT),
              "Abort_op intentional failure; ");
}

// Exit normally.
TEST_F(AbortOpTest, exit_normally) {
  TF_ASSERT_OK(NodeDefBuilder("abort_op", "Abort")
                   .Attr("exit_without_error", true)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  EXPECT_EXIT(RunOpKernel().IgnoreError(), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace tensorflow
