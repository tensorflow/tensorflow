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

#include "tensorflow/compiler/xla/client/lib/triangular_solve.h"

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using TriangularSolveTest = ClientLibraryTestBase;
using TriangularSolveLeftLookingTest = ClientLibraryTestBase;

Array2D<float> AValsLower() {
  return {{2, 0, 0, 0}, {3, 6, 0, 0}, {4, 7, 9, 0}, {5, 8, 10, 11}};
}

Array2D<float> AValsUpper() {
  return {{2, 3, 4, 5}, {0, 6, 7, 8}, {0, 0, 9, 10}, {0, 0, 0, 11}};
}

Array2D<float> BValsRight() {
  return {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
}

Array2D<float> BValsLeft() {
  return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
}

Array2D<complex64> AValsLowerComplex() {
  return {{2, 0, 0, 0},
          {complex64(3, 1), 6, 0, 0},
          {4, complex64(7, 2), 9, 0},
          {5, 8, complex64(10, 3), 11}};
}

Array2D<complex64> AValsUpperComplex() {
  return {{2, 3, complex64(4, 3), 5},
          {0, 6, complex64(7, 2), 8},
          {0, 0, complex64(9, 1), 10},
          {0, 0, 0, 11}};
}

Array2D<complex64> BValsRightComplex() {
  return {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
}

Array2D<complex64> BValsLeftComplex() {
  return {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
}

XLA_TEST_F(TriangularSolveTest, EmptyArrays) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<float>(Array2D<float>(0, 0), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<float>(Array2D<float>(0, 10), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  ComputeAndCompareR2<float>(&builder, Array2D<float>(0, 10),
                             {a_data.get(), b_data.get()});
}

XLA_TEST_F(TriangularSolveTest, SimpleRightLowerTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {0.5, 0.08333334, 0.04629629, 0.03367003},
      {2.5, -0.25, -0.1388889, -0.1010101},
      {4.5, -0.58333331, -0.32407406, -0.23569024},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleRightLowerNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*transpose_a=*/false, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {-0.16414141, -0.06902357, -0.07070707, 0.36363636},
      {0.64393939, 0.06565657, -0.03030303, 0.72727273},
      {1.4520202, 0.2003367, 0.01010101, 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleRightUpperTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/false,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {-0.16414141, -0.06902357, -0.07070707, 0.36363636},
      {0.64393939, 0.06565657, -0.03030303, 0.72727273},
      {1.4520202, 0.2003367, 0.01010101, 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleRightUpperNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsRight(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/false,
                  /*transpose_a=*/false, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {0.5, 0.08333334, 0.04629629, 0.03367003},
      {2.5, -0.25, -0.1388889, -0.1010101},
      {4.5, -0.58333331, -0.32407406, -0.23569024},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftLowerTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {-0.89646465, -0.69444444, -0.49242424},
      {-0.27441077, -0.24074074, -0.20707071},
      {-0.23232323, -0.22222222, -0.21212121},
      {0.90909091, 1., 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftLowerNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*transpose_a=*/false, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftLowerNotransposeIrregularblock) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsLower(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/true,
                  /*transpose_a=*/false, /*conjugate_a=*/false,
                  /*block_size=*/3);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftUpperTranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {0.5, 1.0, 1.5},
      {0.41666667, 0.33333333, 0.25},
      {0.23148148, 0.18518519, 0.13888889},
      {0.16835017, 0.13468013, 0.1010101},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftUpperNotranspose) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data = CreateR2Parameter<float>(AValsUpper(), 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(BValsLeft(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*transpose_a=*/false, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<float> expected({
      {-0.89646465, -0.69444444, -0.49242424},
      {-0.27441077, -0.24074074, -0.20707071},
      {-0.23232323, -0.22222222, -0.21212121},
      {0.90909091, 1., 1.09090909},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleRightLowerTransposeConjugate) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<complex64>(AValsLowerComplex(), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<complex64>(BValsRightComplex(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/false, /*lower=*/true,
                  /*transpose_a=*/true, /*conjugate_a=*/true,
                  /*block_size=*/2);

  Array2D<complex64> expected({
      {0.5, complex64(0.08333333, 0.08333333),
       complex64(0.02777778, -0.0462963), complex64(0.06313131, -0.01094276)},
      {2.5, complex64(-0.25, 0.41666667), complex64(-0.23148148, -0.37962963),
       complex64(0.08670034, -0.02104377)},
      {4.5, complex64(-0.58333333, 0.75), complex64(-0.49074074, -0.71296296),
       complex64(0.11026936, -0.03114478)},
  });

  ComputeAndCompareR2<complex64>(
      &builder, expected, {a_data.get(), b_data.get()}, ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, SimpleLeftUpperTransposeNoconjugate) {
  XlaBuilder builder(TestName());

  XlaOp a, b;
  auto a_data =
      CreateR2Parameter<complex64>(AValsUpperComplex(), 0, "a", &builder, &a);
  auto b_data =
      CreateR2Parameter<complex64>(BValsLeftComplex(), 1, "b", &builder, &b);
  TriangularSolve(a, b,
                  /*left_side=*/true, /*lower=*/false,
                  /*transpose_a=*/true, /*conjugate_a=*/false,
                  /*block_size=*/2);

  Array2D<complex64> expected({
      {0.5, 1., 1.5},
      {0.41666667, 0.33333333, 0.25},
      {complex64(0.20020325, -2.81504065e-01),
       complex64(0.13821138, -4.22764228e-01),
       complex64(0.07621951, -5.64024390e-01)},
      {complex64(0.19678492, 2.55912786e-01),
       complex64(0.17738359, 3.84331116e-01),
       complex64(0.15798226, 5.12749446e-01)},
  });

  ComputeAndCompareR2<complex64>(
      &builder, expected, {a_data.get(), b_data.get()}, ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(TriangularSolveTest, BatchedLeftUpper) {
  XlaBuilder builder(TestName());

  Array3D<float> bvals(7, 5, 5);
  bvals.FillIota(1.);

  // Set avals to the upper triangle of bvals.
  Array3D<float> avals = bvals;
  avals.Each([](absl::Span<const int64> indices, float* value) {
    if (indices[1] > indices[2]) {
      *value = 0;
    }
  });

  XlaOp a, b;
  auto a_data = CreateR3Parameter<float>(avals, 0, "a", &builder, &a);
  auto b_data = CreateR3Parameter<float>(bvals, 1, "b", &builder, &b);
  BatchDot(ConstantR3FromArray3D(&builder, avals),
           TriangularSolve(a, b,
                           /*left_side=*/true, /*lower=*/false,
                           /*transpose_a=*/false, /*conjugate_a=*/false,
                           /*block_size=*/2));

  ComputeAndCompareR3<float>(&builder, bvals, {a_data.get(), b_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

}  // namespace
}  // namespace xla
