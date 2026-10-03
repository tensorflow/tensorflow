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

#include <cstdint>

#include "Eigen/Core"  // from @eigen_archive  // IWYU pragma: keep
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"  // IWYU pragma: keep
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/bfloat16.h"  // IWYU pragma: keep
#include "tsl/platform/bfloat16.h"   // IWYU pragma: keep

namespace tensorflow {
namespace {

class AsStringGraphTest : public OpsTestBase {
 protected:
  absl::Status Init(DataType input_type, const std::string& fill = "",
                    int width = -1, int precision = -1, bool scientific = false,
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

  AddInputFromList<int8_t, int8_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Int64) {
  TF_ASSERT_OK(Init(DT_INT64));

  AddInputFromList<int64_t, int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42", "0", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatDefault) {
  TF_ASSERT_OK(Init(DT_FLOAT));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatScientific) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/true));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-4.200000e+01", "0.000000e+00",
                                        "3.141590e+00", "4.200000e+01"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatShortest) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                    /*scientific=*/false, /*shortest=*/true));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42", "0", "3.14159", "42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatPrecisionOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/2));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", "0.00", "3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FloatWidthOnly) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(
      &expected, {"-42.000000", "0.000000", "3.141590", "42.000000"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Float_5_2_Format) {
  TF_ASSERT_OK(Init(DT_FLOAT, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromList<float, float>(TensorShape({4}), {-42, 0, 3.14159, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({4}));
  test::FillValues<tstring>(&expected, {"-42.00", " 0.00", " 3.14", "42.00"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Complex) {
  TF_ASSERT_OK(Init(DT_COMPLEX64, /*fill=*/"", /*width=*/5, /*precision=*/2));

  AddInputFromList<complex64, complex64>(TensorShape({3}),
                                         {{-4, 2}, {0}, {3.14159, -1}});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(
      &expected, {"(-4.00, 2.00)", "( 0.00, 0.00)", "( 3.14,-1.00)"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Bool) {
  TF_ASSERT_OK(Init(DT_BOOL));

  AddInputFromList<bool, bool>(TensorShape({2}), {true, false});
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

TEST_F(AsStringGraphTest, OnlyOneOfScientificAndShortest) {
  absl::Status s = Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                        /*scientific=*/true, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.message(), "Cannot select both scientific and shortest notation"));
}

TEST_F(AsStringGraphTest, NoShortestForNonFloat) {
  absl::Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                        /*scientific=*/false, /*shortest=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoScientificForNonFloat) {
  absl::Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/-1,
                        /*scientific=*/true);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(
      s.message(),
      "scientific and shortest format not supported for datatype"));
}

TEST_F(AsStringGraphTest, NoPrecisionForNonFloat) {
  absl::Status s = Init(DT_INT32, /*fill=*/"", /*width=*/-1, /*precision=*/5);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(
      absl::StrContains(s.message(), "precision not supported for datatype"));
}

TEST_F(AsStringGraphTest, LongFill) {
  absl::Status s = Init(DT_INT32, /*fill=*/"asdf");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(),
                                "Fill string must be one or fewer characters"));
}

TEST_F(AsStringGraphTest, FillWithZero) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"0", /*width=*/4));

  AddInputFromList<int64_t, int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-042", "0000", "0042"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithSpace) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/" ", /*width=*/4));

  AddInputFromList<int64_t, int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {" -42", "   0", "  42"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar1) {
  TF_ASSERT_OK(Init(DT_INT64, /*fill=*/"-", /*width=*/4));

  AddInputFromList<int64_t, int64_t>(TensorShape({3}), {-42, 0, 42});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({3}));
  test::FillValues<tstring>(&expected, {"-42 ", "0   ", "42  "});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, FillWithChar3) {
  absl::Status s = Init(DT_INT32, /*fill=*/"s");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "Fill argument not supported"));
}

TEST_F(AsStringGraphTest, FillWithChar4) {
  absl::Status s = Init(DT_INT32, /*fill=*/"n");
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "Fill argument not supported"));
}

TEST_F(AsStringGraphTest, Int16) {
  TF_ASSERT_OK(Init(DT_INT16));
  AddInputFromList<int16_t, int16_t>(TensorShape({2}), {-10, 10});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"-10", "10"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Int32) {
  TF_ASSERT_OK(Init(DT_INT32));
  AddInputFromList<int32_t, int32_t>(TensorShape({2}), {-20, 20});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"-20", "20"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Uint8) {
  TF_ASSERT_OK(Init(DT_UINT8));
  AddInputFromList<uint8_t, uint8_t>(TensorShape({2}), {0, 255});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"0", "255"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Uint16) {
  TF_ASSERT_OK(Init(DT_UINT16));
  AddInputFromList<uint16_t, uint16_t>(TensorShape({2}), {0, 65535});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"0", "65535"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Uint32) {
  TF_ASSERT_OK(Init(DT_UINT32));
  AddInputFromList<uint32_t, uint32_t>(TensorShape({2}), {0, 4294967295U});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"0", "4294967295"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Uint64) {
  TF_ASSERT_OK(Init(DT_UINT64));
  AddInputFromList<uint64_t, uint64_t>(TensorShape({2}),
                                       {0, 18446744073709551615ULL});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"0", "18446744073709551615"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Double) {
  TF_ASSERT_OK(Init(DT_DOUBLE, /*fill=*/"", /*width=*/-1, /*precision=*/2));
  AddInputFromList<double, double>(TensorShape({2}), {-3.14159, 2.71828});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"-3.14", "2.72"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Half) {
  TF_ASSERT_OK(Init(DT_HALF, /*fill=*/"", /*width=*/-1, /*precision=*/2));
  AddInputFromList<Eigen::half, Eigen::half>(
      TensorShape({2}),
      {static_cast<Eigen::half>(-1.5f), static_cast<Eigen::half>(1.5f)});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"-1.50", "1.50"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Bfloat16) {
  TF_ASSERT_OK(Init(DT_BFLOAT16, /*fill=*/"", /*width=*/-1, /*precision=*/2));
  AddInputFromList<tsl::bfloat16, tsl::bfloat16>(
      TensorShape({2}),
      {static_cast<tsl::bfloat16>(-2.5f),   // NOLINT(misc-include-cleaner)
       static_cast<tsl::bfloat16>(2.5f)});  // NOLINT(misc-include-cleaner)
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"-2.50", "2.50"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, Complex128) {
  TF_ASSERT_OK(Init(DT_COMPLEX128, /*fill=*/"", /*width=*/5, /*precision=*/2));
  AddInputFromList<complex128, complex128>(TensorShape({2}),
                                           {{-4, 2}, {3.14159, -1}});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"(-4.00, 2.00)", "( 3.14,-1.00)"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, StringWidth) {
  TF_ASSERT_OK(Init(DT_STRING, /*fill=*/"", /*width=*/5));
  AddInputFromList<tstring, tstring>(TensorShape({2}), {"abc", "de"});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_STRING, TensorShape({2}));
  test::FillValues<tstring>(&expected, {"  abc", "   de"});
  test::ExpectTensorEqual<tstring>(expected, *GetOutput(0));
}

TEST_F(AsStringGraphTest, WidthTooLarge) {
  absl::Status s = Init(DT_INT32, /*fill=*/"", /*width=*/200000);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "width must be between -131072 and 131072"));

  s = Init(DT_INT32, /*fill=*/"", /*width=*/-200000);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "width must be between -131072 and 131072"));
}

TEST_F(AsStringGraphTest, PrecisionTooLarge) {
  absl::Status s = Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/200000);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "precision must be between -131072 and 131072"));

  s = Init(DT_FLOAT, /*fill=*/"", /*width=*/-1, /*precision=*/-200000);
  ASSERT_EQ(error::INVALID_ARGUMENT, s.code());
  ASSERT_TRUE(absl::StrContains(s.message(), "precision must be between -131072 and 131072"));
}

}  // end namespace
}  // end namespace tensorflow
