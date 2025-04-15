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

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class TensorScatterUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "TensorScatterUpdate")
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(TensorScatterUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check output.
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(TensorScatterUpdateOpTest, Simple_Two64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64_t>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(TensorScatterUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 1}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 1}), {3});
  AddInputFromArray<float>(TensorShape({1, 1}), {101});
  TF_ASSERT_OK(RunOpKernel());

  // Check output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(TensorScatterUpdateOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 1}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(TensorScatterUpdateOpTest, HigherRank) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3, 1}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  TF_ASSERT_OK(RunOpKernel());

  // Check output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(TensorScatterUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 99, 4});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [99] does not index into shape [5,3]"))
      << s;
}

class TensorScatterUpdateOpErrorOnBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "TensorScatterUpdate")
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Attr("bad_indices_policy", "ERROR")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(TensorScatterUpdateOpErrorOnBadIndicesTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 99, 4});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [99] does not index into shape [5,3]"))
      << s;
}

class TensorScatterUpdateOpIgnoreBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "TensorScatterUpdate")
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Attr("bad_indices_policy", "IGNORE")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(TensorScatterUpdateOpIgnoreBadIndicesTest, DropOutOfRangeIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  // tensor: output tensor of 5x1 shape, initialized to 0.
  AddInputFromArray<float>(TensorShape({5, 1}), {0, 0, 0, 0, 0});
  // Put the bad index in the middle to make sure the others are still updated.
  // Index: [[0], [5], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 5, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  // The valid index range is [0,5). Expect to drop index[1] of value "5" and
  // update otuput[0] and output[2].
  test::FillValues<float>(&expected, {100, 0, 102, 0, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

class ScatterNdUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNdUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// TODO(simister): Re-enable this once binary size is under control.
// TEST_F(ScatterNdUpdateOpTest, Simple_StringType) {
//   MakeOp(DT_STRING_REF, DT_INT32);
//   AddInputFromArray<tstring>(TensorShape({1}), {"Brain"});
//   AddInputFromArray<int32>(TensorShape({1}), {0});
//   AddInputFromArray<tstring>(TensorShape({1}), {"TensorFlow"});
//   TF_ASSERT_OK(RunOpKernel());
//   // Check the new state of the input
//   Tensor params_tensor = *mutable_input(0).tensor;
//   Tensor expected(allocator(), DT_STRING, TensorShape({1}));
//   test::FillValues<tstring>(&expected, {"TensorFlow"});
//   test::ExpectTensorEqual<tstring>(expected, params_tensor);
// }

// TEST_F(ScatterNdUpdateOpTest, Simple_BoolType) {
//   MakeOp(DT_BOOL_REF, DT_INT32);
//   AddInputFromArray<bool>(TensorShape({1}), {false});
//   AddInputFromArray<int32>(TensorShape({1}), {0});
//   AddInputFromArray<bool>(TensorShape({1}), {true});
//   TF_ASSERT_OK(RunOpKernel());
//   // Check the new state of the input
//   Tensor params_tensor = *mutable_input(0).tensor;
//   Tensor expected(allocator(), DT_BOOL, TensorShape({1}));
//   test::FillValues<bool>(&expected, {true});
//   test::ExpectTensorEqual<bool>(expected, params_tensor);
// }

TEST_F(ScatterNdUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Simple_Two64) {
  MakeOp(DT_FLOAT_REF, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64_t>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

/*TEST_F(ScatterNdUpdateOpTest, Simple_ZeroElements) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int32>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Output must not have 0 elements, got shape: "))
      << s;
}*/

TEST_F(ScatterNdUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {3});
  AddInputFromArray<float>(TensorShape({1}), {101});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, HigherRank) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3, 1}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 99, 4});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [99] does not index into shape [5,3]"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_WrongDimsIndices) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 3, 1}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [0,1) of indices[shape=[1,3,1]] = 1 must match dimensions "
      "[0,1) of updates[shape=[3,3]] = 3"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_MismatchedParamsAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(
      TensorShape({3, 4}),
      {100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [1,2) of input[shape=[5,3]] must match dimensions [1,2) of "
      "updates[shape=[3,4]]"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_MismatchedIndicesAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {100, 101, 102, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [0,1) of indices[shape=[3,1]] = 3 must match dimensions [0,1)"
      " of updates[shape=[2,3]] = 2"))
      << s;
}

class ScatterNdUpdateOpErrorOnBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNdUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Attr("bad_indices_policy", "ERROR")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterNdUpdateOpErrorOnBadIndicesTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 99, 4});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [99] does not index into shape [5,3]"))
      << s;
}

class ScatterNdUpdateOpIgnoreBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNdUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Attr("bad_indices_policy", "IGNORE")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterNdUpdateOpIgnoreBadIndicesTest, DropOutOfRangeIndices) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  // Put the bad index in the middle to make sure the others are still updated.
  // ref: output tensor of 5x1 shape, initialized to 0.
  AddInputFromArray<float>(TensorShape({5, 1}), {0, 0, 0, 0, 0});
  // Index: [[0], [5], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 5, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  // The valid index range is [0,5). Expect to drop index[1] of value "5" and
  // update otuput[0] and output[2].
  test::FillValues<float>(&expected, {100, 0, 102, 0, 0});
  test::ExpectTensorEqual<float>(expected, *mutable_input(0).tensor);
}

class ScatterNdUpdateOpConstructionTest : public OpsTestBase {};

TEST_F(ScatterNdUpdateOpConstructionTest, Error_BadIndicesPolicyInvalid) {
  TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNd")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("bad_indices_policy", "AN_UNRECOGNIZED_POLICY")
                   .Finalize(node_def()));
  EXPECT_NE(InitOp(), absl::OkStatus());
}

class ScatterNdOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNd")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterNdOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  // Index: [[0], [4], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  // Shape: output tensor of 5x1 shape.
  AddInputFromArray<int32>(TensorShape({2}), {5, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(ScatterNdOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  // Put the bad index in the middle to make sure the others are still updated.
  // Index: [[0], [5], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 5, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  // Shape: output tensor of 5x1 shape.
  AddInputFromArray<int32>(TensorShape({2}), {5, 1});
  absl::Status s = RunOpKernel();
  // The valid index range is [0,5). Expect "5" to raise error.
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [5] does not index into shape [5,1]"))
      << s;
}

class ScatterNdOpErrorOnBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNd")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(DT_INT32))
                     .Attr("bad_indices_policy", "ERROR")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterNdOpErrorOnBadIndicesTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  // Put the bad index in the middle to make sure the others are still updated.
  // Index: [[0], [5], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 5, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  // Shape: output tensor of 5x1 shape.
  AddInputFromArray<int32>(TensorShape({2}), {5, 1});
  absl::Status s = RunOpKernel();
  // The valid index range is [0,5). Expect "5" to raise error.
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[1] = [5] does not index into shape [5,1]"))
      << s;
}

class ScatterNdOpIgnoreBadIndicesTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNd")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(variable_type))
                     .Input(FakeInput(DT_INT32))
                     .Attr("bad_indices_policy", "IGNORE")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterNdOpIgnoreBadIndicesTest, DropOutOfRangeIndices) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  // Put the bad index in the middle to make sure the others are still updated.
  // Index: [[0], [5], [2]].
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 5, 2});
  // Updates: [100, 101, 102].
  AddInputFromArray<float>(TensorShape({3, 1}), {100, 101, 102});
  // Shape: output tensor of 5x1 shape.
  AddInputFromArray<int32>(TensorShape({2}), {5, 1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 1}));
  // The valid index range is [0,5). Expect to drop index[1] of value "5" and
  // update otuput[0] and output[2].
  test::FillValues<float>(&expected, {100, 0, 102, 0, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

class ScatterNdOpConstructionTest : public OpsTestBase {};

TEST_F(ScatterNdOpConstructionTest, Error_BadIndicesPolicyInvalid) {
  TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNd")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("bad_indices_policy", "AN_UNRECOGNIZED_POLICY")
                   .Finalize(node_def()));
  EXPECT_NE(InitOp(), absl::OkStatus());
}

class ScatterNdUpdateBM : public ScatterNdUpdateOpTest {
 public:
  void TestBody() override {}
  void MakeBenchmarkOp(const char* op, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", op)
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }
};

template <typename Index>
void BM_ScatterNdHelper(::testing::benchmark::State& state, int embedding_size,
                        const char* op) {
  const int kRows = 10000000 / embedding_size;
  std::vector<float> values;
  values.reserve(kRows);
  for (int i = 0; i < kRows * embedding_size; i++) {
    values.push_back(i);
  }
  const int kNumUpdates = 1000;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices;
  std::vector<float> updates;
  for (int i = 0; i < kNumUpdates; i++) {
    indices.push_back(rnd.Uniform(kRows));
    for (int j = 0; j < embedding_size; j++) {
      updates.push_back(i * 10 + j);
    }
  }

  ScatterNdUpdateBM bm;
  bm.MakeBenchmarkOp(op, DataTypeToEnum<Index>::v());
  bm.AddInputFromArray<float>(TensorShape({kRows, embedding_size}), values);
  bm.AddInputFromArray<Index>(TensorShape({kNumUpdates}), indices);
  bm.AddInputFromArray<float>(TensorShape({kNumUpdates, embedding_size}),
                              updates);
  for (auto i : state) {
    absl::Status s = bm.RunOpKernel();
  }
  state.SetItemsProcessed((static_cast<int64_t>(kNumUpdates) * embedding_size) *
                          state.iterations());
}

void BM_ScatterNdUpdateInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int32>(state, embedding_size, "ScatterNdUpdate");
}
void BM_ScatterNdUpdateInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int64_t>(state, embedding_size, "ScatterNdUpdate");
}

void BM_ScatterNdAddInt32(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int32>(state, embedding_size, "ScatterNdAdd");
}
void BM_ScatterNdAddInt64(::testing::benchmark::State& state) {
  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int64_t>(state, embedding_size, "ScatterNdAdd");
}

BENCHMARK(BM_ScatterNdUpdateInt32)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);
BENCHMARK(BM_ScatterNdUpdateInt64)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

BENCHMARK(BM_ScatterNdAddInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterNdAddInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

}  // namespace
}  // namespace tensorflow
