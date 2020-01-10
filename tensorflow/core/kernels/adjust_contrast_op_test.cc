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

class AdjustContrastOpTest : public OpsTestBase {
 protected:
  void MakeOp() { RequireDefaultOps(); }
};

TEST_F(AdjustContrastOpTest, Simple_1113) {
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 1, 1, 3}), {-1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {2.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 3}));
  test::FillValues<float>(&expected, {0, 2, 2});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Simple_1223) {
  RequireDefaultOps();
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2, 2, 3}),
                           {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
  AddInputFromArray<float>(TensorShape({}), {0.2});
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {10.0});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 3}));
  test::FillValues<float>(
      &expected, {2.2, 6.2, 10, 2.4, 6.4, 10, 2.6, 6.6, 10, 2.8, 6.8, 10});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AdjustContrastOpTest, Big_99x99x3) {
  EXPECT_OK(NodeDefBuilder("adjust_constrast_op", "AdjustContrast")
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Input(FakeInput(DT_FLOAT))
                .Attr("T", DT_FLOAT)
                .Finalize(node_def()));
  EXPECT_OK(InitOp());

  std::vector<float> values;
  for (int i = 0; i < 99 * 99 * 3; ++i) {
    values.push_back(i % 255);
  }

  AddInputFromArray<float>(TensorShape({1, 99, 99, 3}), values);
  AddInputFromArray<float>(TensorShape({}), {0.2});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255});
  ASSERT_OK(RunOpKernel());
}

}  // namespace tensorflow
