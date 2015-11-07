#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/tensor.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace {

class ReverseOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "Reverse")
                  .Input(FakeInput(data_type))
                  .Input(FakeInput())
                  .Attr("T", data_type)
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(ReverseOpTest, Reverse_0) {
  MakeOp(DT_FLOAT);
  AddInputFromArray<float>(TensorShape({}), {3});
  AddInputFromArray<bool>(TensorShape({}), {true});
  ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
  expected.scalar<float>() = expected.scalar<float>().constant(3.f);
  test::ExpectTensorEqual<float>(expected, *output);
}

TEST_F(ReverseOpTest, Reverse_234) {
  MakeOp(DT_FLOAT);

  // Feed and run
  // [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
  //  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
  AddInputFromArray<float>(TensorShape({2, 3, 4}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19, 20, 21, 22, 23});
  AddInputFromArray<bool>(TensorShape({3}), {true, false, true});

  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor* params_tensor = GetOutput(0);
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3, 4}));
  // Should become
  // [[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
  //  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]
  test::FillValues<float>(
      &expected, {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3, 2, 1, 0, 7,
                  6, 5, 4, 11, 10, 9, 8});
  test::ExpectTensorEqual<float>(expected, *params_tensor);
}

TEST_F(ReverseOpTest, Reverse_1234) {
  MakeOp(DT_FLOAT);

  // Feed and run
  // [[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
  //   [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]]
  AddInputFromArray<float>(TensorShape({1, 2, 3, 4}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19, 20, 21, 22, 23});
  AddInputFromArray<bool>(TensorShape({4}), {true, true, false, true});

  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor* params_tensor = GetOutput(0);
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 3, 4}));
  // Should become
  // [[[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
  //   [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]]
  test::FillValues<float>(
      &expected, {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3, 2, 1, 0, 7,
                  6, 5, 4, 11, 10, 9, 8});
  test::ExpectTensorEqual<float>(expected, *params_tensor);
}

}  // namespace
}  // namespace tensorflow
