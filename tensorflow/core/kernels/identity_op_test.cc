#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace {

class IdentityOpTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type) {
    RequireDefaultOps();
    TF_CHECK_OK(NodeDefBuilder("op", "Identity")
                    .Input(FakeInput(input_type))
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(IdentityOpTest, Int32Success_6) {
  ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(IdentityOpTest, Int32Success_2_3) {
  ASSERT_OK(Init(DT_INT32));
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({2, 3}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(IdentityOpTest, StringSuccess) {
  ASSERT_OK(Init(DT_STRING));
  AddInputFromArray<string>(TensorShape({6}), {"A", "b", "C", "d", "E", "f"});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({6}));
  test::FillValues<string>(&expected, {"A", "b", "C", "d", "E", "f"});
  test::ExpectTensorEqual<string>(expected, *GetOutput(0));
}

TEST_F(IdentityOpTest, RefInputError) { ASSERT_OK(Init(DT_INT32_REF)); }

}  // namespace
}  // namespace tensorflow
