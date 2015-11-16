// TODO(shlens, sherrym): Consider adding additional tests in image_ops.py in
// order to compare the reference implementation for image resizing in Python
// Image Library.
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

class ResizeNearestNeighborOpTest : public OpsTestBase {
 protected:
  ResizeNearestNeighborOpTest() {
    RequireDefaultOps();
    EXPECT_OK(NodeDefBuilder("resize_nn", "ResizeNearestNeighbor")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT32))
                  .Finalize(node_def()));
    EXPECT_OK(InitOp());
  }
};

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2To1x1) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {1, 1});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 1, 1}));

  // clang-format off
  test::FillValues<float>(&expected, {1});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2To3x3) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 3, 3, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2,
     1, 1, 2,
     3, 3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2To2x5) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {2, 5});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 5, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1, 2, 2,
     3, 3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2To5x2) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {5, 2});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 5, 2, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 2,
     1, 2,
     1, 2,
     3, 4,
     3, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2To4x4) {
  // Input:
  //  1, 2
  //  3, 4
  AddInputFromArray<float>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({2}), {4, 4});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 4, 4, 1}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 2, 2,
     1, 1, 2, 2,
     3, 3, 4, 4,
     3, 3, 4, 4});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ResizeNearestNeighborOpTest, TestNearest2x2x2x2To2x3x3x2) {
  // Input:
  //  [ [ 1, 1 ], [ 2, 2],
  //    [ 3, 3 ], [ 4, 4] ],
  //  [ [ 5, 5 ], [ 6, 6],
  //    [ 7, 7 ], [ 8, 8] ]
  AddInputFromArray<float>(TensorShape({2, 2, 2, 2}),
                           {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
  AddInputFromArray<int32>(TensorShape({2}), {3, 3});
  ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 3, 2}));

  // clang-format off
  test::FillValues<float>(&expected,
    {1, 1, 1,
     1, 2, 2,
     1, 1, 1,
     1, 2, 2,
     3, 3, 3,
     3, 4, 4,
     5, 5, 5,
     5, 6, 6,
     5, 5, 5,
     5, 6, 6,
     7, 7, 7,
     7, 8, 8});

  // clang-format on
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace tensorflow
