#include "tensorflow/core/framework/fake_input.h"
#include <gtest/gtest.h>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

// Tests for the switch op
class SwitchOpTest : public OpsTestBase {
 protected:
  void Initialize(DataType dt) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("op", "Switch")
                  .Input(FakeInput(dt))
                  .Input(FakeInput())
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(SwitchOpTest, Int32Success_6_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, Int32Success_6_s1) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {true});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

TEST_F(SwitchOpTest, Int32Success_2_3_s0) {
  Initialize(DT_INT32);
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<bool>(TensorShape({}), {false});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
  EXPECT_EQ(nullptr, GetOutput(1));
}

TEST_F(SwitchOpTest, StringSuccess_s1) {
  Initialize(DT_STRING);
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  AddInputFromArray<bool>(TensorShape({}), {true});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(1));
  EXPECT_EQ(nullptr, GetOutput(0));
}

}  // namespace
}  // namespace tensorflow
