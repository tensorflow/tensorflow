#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

class PrintingGraphTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type1, DataType input_type2, string msg = "",
              int first_n = -1, int summarize = 3) {
    RequireDefaultOps();
    TF_CHECK_OK(NodeDefBuilder("op", "Print")
                    .Input(FakeInput(input_type1))
                    .Input(FakeInput(2, input_type2))
                    .Attr("message", msg)
                    .Attr("first_n", first_n)
                    .Attr("summarize", summarize)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(PrintingGraphTest, Int32Success_6) {
  ASSERT_OK(Init(DT_INT32, DT_INT32));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, Int32Success_Summarize6) {
  ASSERT_OK(Init(DT_INT32, DT_INT32, "", -1, 6));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, StringSuccess) {
  ASSERT_OK(Init(DT_INT32, DT_STRING));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<string>(TensorShape({}), {"foo"});
  AddInputFromArray<string>(TensorShape({}), {"bar"});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, MsgSuccess) {
  ASSERT_OK(Init(DT_INT32, DT_STRING, "Message: "));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<string>(TensorShape({}), {"foo"});
  AddInputFromArray<string>(TensorShape({}), {"bar"});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

TEST_F(PrintingGraphTest, FirstNSuccess) {
  ASSERT_OK(Init(DT_INT32, DT_STRING, "", 3));
  AddInputFromArray<int32>(TensorShape({6}), {1, 2, 3, 4, 5, 6});
  AddInputFromArray<string>(TensorShape({}), {"foo"});
  AddInputFromArray<string>(TensorShape({}), {"bar"});
  // run 4 times but we only print 3 as intended
  for (int i = 0; i < 4; i++) ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_INT32, TensorShape({6}));
  test::FillValues<int32>(&expected, {1, 2, 3, 4, 5, 6});
  test::ExpectTensorEqual<int32>(expected, *GetOutput(0));
}

}  // end namespace
}  // end namespace tensorflow
