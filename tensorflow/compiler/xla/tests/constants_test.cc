/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Tests that constants in program memory round trip as expected.

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConstantsTest : public ClientLibraryTestBase {
 protected:
  const ErrorSpec error_spec_{1e-3, 1e-5};
};

TEST_F(ConstantsTest, ZeroCellF32) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>({});

  ComputeAndCompareR1<float>(&builder, {}, {}, error_spec_);
}

TEST_F(ConstantsTest, OneCellF32) {
  std::vector<float> constant = {2.0};

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>(constant);

  ComputeAndCompareR1<float>(&builder, constant, {}, error_spec_);
}

TEST_F(ConstantsTest, OneCellS32) {
  std::vector<int32> constant = {2};

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<int32>(constant);

  ComputeAndCompareR1<int32>(&builder, constant, {});
}

TEST_F(ConstantsTest, OneCellU32) {
  std::vector<uint32> constant = {2};

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<uint32>(constant);

  ComputeAndCompareR1<uint32>(&builder, constant, {});
}

TEST_F(ConstantsTest, EightCells) {
  std::vector<float> constant = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>(constant);

  ComputeAndCompareR1<float>(&builder, constant, {}, error_spec_);
}

TEST_F(ConstantsTest, SixteenCells) {
  std::vector<float> constant = {0.0, 1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR1<float>(constant);

  ComputeAndCompareR1<float>(&builder, constant, {}, error_spec_);
}

TEST_F(ConstantsTest, Empty_0x2) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantR2FromArray2D<float>(Array2D<float>(0, 2));

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 2), {}, error_spec_);
}

TEST_F(ConstantsTest, Small_2x2) {
  std::unique_ptr<Array2D<float>> constant =
      MakeLinspaceArray2D(100.0, 200.0, 2, 2);

  ComputationBuilder builder(client_, TestName());
  builder.ConstantR2FromArray2D<float>(*constant);

  ComputeAndCompareR2<float>(&builder, *constant, {}, error_spec_);
}

TEST_F(ConstantsTest, Empty_3x0x2) {
  ComputationBuilder builder(client_, TestName());
  auto constant = builder.ConstantLiteral(
      *Literal::CreateR3FromArray3D<float>(Array3D<float>(3, 0, 2)));

  ComputeAndCompareR3<float>(&builder, Array3D<float>(3, 0, 2), {});
}

TEST_F(ConstantsTest, Small_2x2x2) {
  ComputationBuilder builder(client_, TestName());
  Array3D<float> array3d({
      // x0  x1
      {{1.f, 2.f},   // y0
       {3.f, 4.f}},  // y1

      {{5.f, 6.f},   // y0
       {7.f, 8.f}},  // y1
  });
  auto constant =
      builder.ConstantLiteral(*Literal::CreateR3FromArray3D<float>(array3d));

  ComputeAndCompareR3<float>(&builder, array3d, {});
}

TEST_F(ConstantsTest, Small_3x2x1x1) {
  Array4D<float> input_array(3, 2, 1, 1);
  Array2D<float> pz({
      // z0 z1
      {-1.0f, 4.1f},  // p0
      {2.0f, 4.1f},   // p1
      {5.0f, 4.4f},   // p2
  });
  input_array.FillWithPZ(pz);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR4FromArray4D(input_array);

  {
    ComputationBuilder builder(client_, TestName());
    builder.ConstantLiteral(*input_literal);
    ComputeAndCompareR4<float>(&builder, input_array, {}, error_spec_);
  }

  {
    ComputationBuilder builder(client_, TestName());
    builder.ConstantR4FromArray4D<float>(input_array);
    ComputeAndCompareR4<float>(&builder, input_array, {}, error_spec_);
  }
}

// TODO(b/29263943): Support tuple constants.
TEST_F(ConstantsTest, DISABLED_TupleConstant) {
  ComputationBuilder builder(client_, TestName());
  builder.ConstantLiteral(
      *Literal::MakeTuple({Literal::CreateR2<float>({{1.0}, {2.0}}).get(),
                           Literal::CreateR1<float>({2.0, 42}).get()}));

  std::unique_ptr<Literal> result = ExecuteAndTransferOrDie(&builder, {});

  LiteralTestUtil::ExpectR2Near<float>(
      {{1.0}, {2.0}}, LiteralView::Create(*result, {0}), error_spec_);
  LiteralTestUtil::ExpectR1Near<float>(
      {2.0, 42.0}, LiteralView::Create(*result, {1}), error_spec_);
}

}  // namespace
}  // namespace xla
