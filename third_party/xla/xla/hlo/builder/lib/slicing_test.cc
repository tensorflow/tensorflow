/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/slicing.h"

#include <cstdint>

#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using SlicingTest = xla::ClientLibraryTestBase;

xla::Array2D<float> BValsRight() {
  return {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
}

xla::Array2D<float> BValsLeft() {
  return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
}

xla::Array2D<float> AValsFull() {
  return {{2, 0, 1, 2}, {3, 6, 0, 1}, {4, 7, 9, 0}, {5, 8, 10, 11}};
}

xla::Array3D<float> BatchedAValsFull() {
  return {{
              {2, 0, 1, 2},
              {3, 6, 0, 1},
              {4, 7, 9, 0},
              {5, 8, 10, 11},
          },
          {
              {16, 24, 8, 12},
              {24, 61, 82, 48},
              {8, 82, 456, 106},
              {12, 48, 106, 62},
          }};
}

TEST_F(SlicingTest, Simple2dLookup) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, x, y;
  auto a_data = CreateR2Parameter<float>(BValsRight(), 0, "a", &builder, &a);
  auto x_data = CreateR0Parameter<int>(2, 1, "x", &builder, &x);
  auto y_data = CreateR0Parameter<int>(1, 2, "y", &builder, &y);
  DynamicSliceInMinorDims(a, {x, y}, {1, 1});

  ComputeAndCompareR2<float>(&builder, {{10}},
                             {a_data.get(), x_data.get(), y_data.get()},
                             xla::ErrorSpec(1e-2, 1e-2));
}

TEST_F(SlicingTest, Simple3dLookup) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto index_data = CreateR0Parameter<int>(1, 1, "index", &builder, &index);

  DynamicSliceInMinorDims(a, {index, xla::ConstantR0<int32_t>(&builder, 0)},
                          {1, 4});

  ComputeAndCompareR3<float>(&builder, {{{3, 6, 0, 1}}, {{24, 61, 82, 48}}},
                             {a_data.get(), index_data.get()});
}

TEST_F(SlicingTest, NestedLookup) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto index_data = CreateR0Parameter<int>(1, 1, "index", &builder, &index);

  auto slice = DynamicSliceInMinorDims(
      a, {index, xla::ConstantR0<int32_t>(&builder, 0)}, {1, 4});
  DynamicSliceInMinorDims(slice, {xla::ConstantR0<int32_t>(&builder, 0), index},
                          {1, 1});

  ComputeAndCompareR3<float>(&builder, {{{6}}, {{61}}},
                             {a_data.get(), index_data.get()});
}

TEST_F(SlicingTest, SimpleSliceUpdate) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, b, x, y;
  auto a_data = CreateR2Parameter<float>(AValsFull(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>({{9, 1, -10}}, 1, "b", &builder, &b);
  auto x_data = CreateR0Parameter<int>(2, 2, "x", &builder, &x);
  auto y_data = CreateR0Parameter<int>(1, 3, "y", &builder, &y);

  DynamicUpdateSliceInMinorDims(a, b, {x, y});

  xla::Array2D<float> expected(
      {{{2, 0, 1, 2}, {3, 6, 0, 1}, {4, 9, 1, -10}, {5, 8, 10, 11}}});

  ComputeAndCompareR2<float>(
      &builder, expected,
      {a_data.get(), b_data.get(), x_data.get(), y_data.get()});
}

TEST_F(SlicingTest, NestedSliceUpdate) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, b, x, y;
  auto a_data = CreateR2Parameter<float>(AValsFull(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>({{1, -10}}, 1, "b", &builder, &b);
  auto x_data = CreateR0Parameter<int>(2, 2, "x", &builder, &x);
  auto y_data = CreateR0Parameter<int>(1, 3, "y", &builder, &y);

  auto z = xla::ConstantR0<int32_t>(&builder, 0);
  auto slice = DynamicSliceInMinorDims(a, {x, z}, {1, 4});
  auto inner = DynamicUpdateSliceInMinorDims(slice, b, {z, y});
  DynamicUpdateSlice(a, inner, {x, z});

  xla::Array2D<float> expected(
      {{{2, 0, 1, 2}, {3, 6, 0, 1}, {4, 1, -10, 0}, {5, 8, 10, 11}}});

  ComputeAndCompareR2<float>(
      &builder, expected,
      {a_data.get(), b_data.get(), x_data.get(), y_data.get()});
}

TEST_F(SlicingTest, TorchGatherSparse) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR2Parameter<int>({{1, 2}, {3, 4}}, 0, "input", &builder, &input);
  auto index_data =
      CreateR2Parameter<int>({{0, 0}, {1, 0}}, 1, "index", &builder, &index);
  TorchGather(input, index, 1);

  ComputeAndCompareR2<int>(&builder, {{1, 1}, {4, 3}},
                           {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, TorchGatherDense) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR2Parameter<int>({{1, 2}, {3, 4}}, 0, "input", &builder, &input);
  auto index_data =
      CreateR2Parameter<int>({{0, 0}, {1, 0}}, 1, "index", &builder, &index);
  TorchGather(input, index, 1, false);

  ComputeAndCompareR2<int>(&builder, {{1, 1}, {4, 3}},
                           {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, TorchScatterDense) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp src, index, input;
  auto input_data = CreateR2Parameter<int>({{0, 0, 0}, {0, 0, 0}}, 0, "input",
                                           &builder, &input);
  auto index_data =
      CreateR2Parameter<int>({{1, 0}, {1, 2}}, 1, "index", &builder, &index);
  auto src_data =
      CreateR2Parameter<int>({{1, 2}, {3, 4}}, 2, "src", &builder, &src);
  TorchScatterDense(input, index, src, 1,
                    [](XlaOp l, XlaOp r) { return l + r; });

  ComputeAndCompareR2<int>(
      &builder, {{2, 1, 0}, {0, 3, 4}},
      {input_data.get(), index_data.get(), src_data.get()});
}

TEST_F(SlicingTest, TorchIndexSelectOn0) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR2Parameter<float>({{0.1427, 0.0231, -0.5414, -1.0009},
                                {-0.4664, 0.2647, -0.1228, -1.1068},
                                {-1.1734, -0.6571, 0.7230, -0.6004}},
                               0, "input", &builder, &input);
  auto index_data =
      CreateR1Parameter<int>({0, 2}, 1, "index", &builder, &index);
  TorchIndexSelect(input, index, 0);

  ComputeAndCompareR2<float>(
      &builder,
      {{0.1427, 0.0231, -0.5414, -1.0009}, {-1.1734, -0.6571, 0.7230, -0.6004}},
      {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, TorchIndexSelectOn0Size1) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data = CreateR2Parameter<float>(
      {{-1.1734, -0.6571, 0.7230, -0.6004}}, 0, "input", &builder, &input);
  auto index_data =
      CreateR1Parameter<int>({0, 0, 0, 0, 0, 0}, 1, "index", &builder, &index);
  TorchIndexSelect(input, index, 0);

  ComputeAndCompareR2<float>(&builder,
                             {{-1.1734, -0.6571, 0.7230, -0.6004},
                              {-1.1734, -0.6571, 0.7230, -0.6004},
                              {-1.1734, -0.6571, 0.7230, -0.6004},
                              {-1.1734, -0.6571, 0.7230, -0.6004},
                              {-1.1734, -0.6571, 0.7230, -0.6004},
                              {-1.1734, -0.6571, 0.7230, -0.6004}},
                             {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, TorchIndexSelectOn1) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR2Parameter<float>({{0.1427, 0.0231, -0.5414, -1.0009},
                                {-0.4664, 0.2647, -0.1228, -1.1068},
                                {-1.1734, -0.6571, 0.7230, -0.6004}},
                               0, "input", &builder, &input);
  auto index_data =
      CreateR1Parameter<int>({0, 2}, 1, "index", &builder, &index);

  TorchIndexSelect(input, index, 1);

  ComputeAndCompareR2<float>(
      &builder, {{0.1427, -0.5414}, {-0.4664, -0.1228}, {-1.1734, 0.7230}},
      {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, EmptyIndexSelect) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR2Parameter<float>({{0}, {0}, {0}}, 0, "input", &builder, &input);
  auto index_data = CreateR1Parameter<int>({}, 1, "index", &builder, &index);
  TorchIndexSelect(input, index, 1);
  ComputeAndCompareR2<float>(&builder, {{}, {}, {}},
                             {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, DoubleEmptyIndexSelect) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  Literal l(ShapeUtil::MakeShape(F32, {0, 1, 2, 0}));
  Literal i(ShapeUtil::MakeShape(S32, {0}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data,
      CreateParameterAndTransferLiteral(0, l, "input", &builder, &input));
  TF_ASSERT_OK_AND_ASSIGN(
      auto index_data,
      CreateParameterAndTransferLiteral(1, i, "index", &builder, &index));
  TorchIndexSelect(input, index, 0);
  ComputeAndCompareLiteral(&builder, l, {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, EmptyIndexSelectNonZero) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  Literal l(ShapeUtil::MakeShape(F32, {0, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto input_data,
      CreateParameterAndTransferLiteral(0, l, "input", &builder, &input));
  auto index_data =
      CreateR1Parameter<int>({0, 0, 0}, 1, "index", &builder, &index);
  TorchIndexSelect(input, index, 0);
  ComputeAndCompareR2<float>(&builder,
                             {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
                             {input_data.get(), index_data.get()});
}

TEST_F(SlicingTest, BatchTorchIndexSelectOn0) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp input, index;
  auto input_data =
      CreateR3Parameter<int>({{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
                              {{3, 2, 1, 0}, {7, 6, 5, 4}, {11, 10, 9, 8}}},
                             0, "input", &builder, &input);
  auto index_data =
      CreateR2Parameter<int>({{0, 2}, {1, 2}}, 1, "index", &builder, &index);
  TorchIndexSelect(input, index, 1, 1);

  ComputeAndCompareR3<int>(
      &builder,
      {{{0, 1, 2, 3}, {8, 9, 10, 11}}, {{7, 6, 5, 4}, {11, 10, 9, 8}}},
      {input_data.get(), index_data.get()});
}

}  // namespace
}  // namespace xla
