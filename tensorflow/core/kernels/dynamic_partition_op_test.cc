#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class DynamicPartitionOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "DynamicPartition")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT32))
                  .Attr("num_partitions", 4)
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(DynamicPartitionOpTest, Simple_OneD) {
  MakeOp();

  // Similar to how we would use this to split embedding ids to be looked up

  // Feed and run
  AddInputFromArray<float>(TensorShape({6}), {0, 13, 2, 39, 4, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 3, 2, 1});
  ASSERT_OK(RunOpKernel());

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&expected, {0, 13});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
    test::FillValues<float>(&expected, {17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&expected, {2, 4});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
    test::FillValues<float>(&expected, {39});
    test::ExpectTensorEqual<float>(expected, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, Simple_TwoD) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(
      TensorShape({6, 3}),
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 3, 2, 1});
  ASSERT_OK(RunOpKernel());

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
    test::FillValues<float>(&expected, {0, 1, 2, 3, 4, 5});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3}));
    test::FillValues<float>(&expected, {15, 16, 17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
    test::FillValues<float>(&expected, {6, 7, 8, 12, 13, 14});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3}));
    test::FillValues<float>(&expected, {9, 10, 11});
    test::ExpectTensorEqual<float>(expected, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, SomeOutputsEmpty) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(TensorShape({6}), {0, 13, 2, 39, 4, 17});
  AddInputFromArray<int32>(TensorShape({6}), {0, 0, 2, 2, 0, 2});
  ASSERT_OK(RunOpKernel());

  TensorShape empty_one_dim;
  empty_one_dim.AddDim(0);
  Tensor expected_empty(allocator(), DT_FLOAT, empty_one_dim);

  // Check the output sizes
  {  // Output 0
    Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected, {0, 13, 4});
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }
  {  // Output 1
    test::ExpectTensorEqual<float>(expected_empty, *GetOutput(1));
  }
  {  // Output 2
    Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected, {2, 39, 17});
    test::ExpectTensorEqual<float>(expected, *GetOutput(2));
  }
  {  // Output 3
    test::ExpectTensorEqual<float>(expected_empty, *GetOutput(3));
  }
}

TEST_F(DynamicPartitionOpTest, Error_IndexOutOfRange) {
  MakeOp();

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({5}), {0, 2, 99, 2, 2});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString()).contains("partitions[2] = 99 is not in [0, 4)"))
      << s;
}

}  // namespace
}  // namespace tensorflow
