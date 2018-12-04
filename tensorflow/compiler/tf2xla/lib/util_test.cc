/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/util.h"

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using UtilTest = xla::ClientLibraryTestBase;
using UtilLeftLookingTest = xla::ClientLibraryTestBase;

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

XLA_TEST_F(UtilTest, Simple2dLookup) {
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

XLA_TEST_F(UtilTest, Simple3dLookup) {
  xla::XlaBuilder builder(TestName());

  xla::XlaOp a, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto index_data = CreateR0Parameter<int>(1, 1, "index", &builder, &index);

  DynamicSliceInMinorDims(a, {index, xla::ConstantR0<int32>(&builder, 0)},
                          {1, 4});

  ComputeAndCompareR3<float>(&builder, {{{3, 6, 0, 1}}, {{24, 61, 82, 48}}},
                             {a_data.get(), index_data.get()});
}

XLA_TEST_F(UtilTest, SimpleSliceUpdate) {
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

XLA_TEST_F(UtilTest, RowBatchDot) {
  xla::XlaBuilder builder(TestName());

  int n = 4;

  xla::XlaOp a, row, index;
  auto a_data =
      CreateR3Parameter<float>(BatchedAValsFull(), 0, "a", &builder, &a);
  auto row_data = CreateR3Parameter<float>({{{9, 1, 0, 0}}, {{2, 4, 0, 0}}}, 1,
                                           "row", &builder, &row);
  // Select {{3, 6, 0, 1}, {24, 61,  82,  48}} out of BatchedAValsFull().
  auto index_data = CreateR0Parameter<int>(1, 2, "index", &builder, &index);

  auto l_index = DynamicSliceInMinorDims(
      a, {index, xla::ConstantR0<int32>(&builder, 0)}, {1, n});
  BatchDot(l_index, TransposeInMinorDims(row));

  ComputeAndCompareR3<float>(&builder, {{{33}}, {{292}}},
                             {a_data.get(), row_data.get(), index_data.get()});
}

}  // namespace
}  // namespace tensorflow
