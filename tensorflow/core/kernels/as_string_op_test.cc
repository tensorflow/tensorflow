/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

class AsStringGraphTest : public OpsTestBase {
 protected:
  Status Init(DataType input_type, const string& fill = "", int width = -1,
              int precision = -1, bool scientific = false,
              bool shortest = false) {
    TF_CHECK_OK(NodeDefBuilder("op", "AsString")
                    .Input(FakeInput(input_type))
                    .Attr("fill", fill)
                    .Attr("precision", precision)
                    .Attr("scientific", scientific)
                    .Attr("shortest", shortest)
                    .Attr("width", width)
                    .Finalize(node_def()));
    return InitOp();
  }
};

TEST_F(AsStringGraphTest, Int8) {
  TF_ASSERT_OK(Init(DT_INT8));

  AddInputFromArray<int8>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Int64) {
  TF_ASSERT_OK(Init(DT_INT64));

  AddInputFromArray<int64>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatDefault) {
  TF_ASSERT_OK(Init(DT_FLOAT));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatScientific) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/true));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-4.200000e+01", "0.000000e+00",
                                        "3.141590e+00", "4.200000e+01"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatShortest) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/false, /*shortest=*/true));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42", "0", "3.14159", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatPrecisionOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/2));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", "0.00", "3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatWidthOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Float_5_2_Format) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromArray<float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", " 0.00", " 3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Complex) {
  TF_ASSERT_OK(Init(DT_COMPLEX64, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromArray<complex64>(TensorShape({3}), {{-4, 2}, {0}, {3.14159, -1}});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(
      &expected, {"(-4.00, 2.00)", "( 0.00, 0.00)", "( 3.14,-1.00)"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Bool) {
  TF_ASSERT_OK(Init(DT_BOOL));

  AddInputFromArray<bool>(TensorShape({2}), {true, false});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"true", "false"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Variant) {
  TF_ASSERT_OK(Init(DT_VARIANT));

  AddInput(DT_VARIANT, TensorShape({4}));
  auto inputs = mutable_input(0)->flat<Variant>();
  inputs(0) = 2;
  inputs(1) = 3;
  inputs(2) = true;
  inputs(3) = Tensor("hi");
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"Variant<type: int value: 2>", "Variant<type: int value: 3>",
                  "Variant<type: bool value: 1>",
                  ("Variant<type: tensorflow::Tensor value: Tensor<type: string"
                   " shape: [] values: hi>>")});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, String) {
  Status s = Init(DT_STRING);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "Value for attr 'T' of string is not in the list of allowed values"));
}

TEST_F(AsStringGraphTest, OnlyOneOfScientificAndShortest) {
  Status s = Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/true, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(),
                        "Cannot select both scientific and shortest notation"));
}

TEST_F(AsStringGraphTest, NoShortestForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/false, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoScientificForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                  /*scientific=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.error_message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoPrecisionForNonFloat) {
  Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/5);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.error_message(),
                                "precision not supported for datatype"));
}

TEST_F(AsStringGraphTest, LongFill) {
  Status s = Init(DT_INT32, /*fill=*/"asdf");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.error_message(),
                                "Fill string must be one or fewer characters"));
}

TEST_F(AsStringGraphTest, FillWithZero) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"0", /*width=*/4));

  AddInputFromArray<int64>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-042", "0000", "0042"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithSpace) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/" ", /*width=*/4));

  AddInputFromArray<int64>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {" -42", "   0", "  42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar1) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"-", /*width=*/4));

  AddInputFromArray<int64>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42 ", "0   ", "42  "});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar3) {
  Status s = Init(DT_INT32, /*fill=*/"s");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(), "Fill argument not supported"));
}

TEST_F(AsStringGraphTest, FillWithChar4) {
  Status s = Init(DT_INT32, /*fill=*/"n");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.error_message(), "Fill argument not supported"));
}

}  // end namespace
}  // end namespace tensorflow
