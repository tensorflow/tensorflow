/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/self_adjoint_eigen.h"

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {

class SelfAdjointEigenTest : public ClientLibraryTestBase {
 protected:
  void SetUp() override {
    ClientLibraryTestBase::SetUp();
    batch_3d_4x4_ = Array3D<float>{
        {
            {4, 6, 8, 10},
            {6, 45, 54, 63},
            {8, 54, 146, 166},
            {10, 63, 166, 310},
        },
        {
            {16, 24, 8, 12},
            {24, 61, 82, 48},
            {8, 82, 100, 6},
            {12, 48, 6, 62},
        },
    };
    matrix2d_8x8_ = Array2D<float>{
        {14., 123., 49., 112., 115., 173., 182., 125.},
        {123., 14., 60., 118., 150., 130., 91., 72.},
        {49., 60., 138., 111., 106., 101., 115., 142.},
        {112., 118., 111., 142., 91., 130., 25., 61.},
        {115., 150., 106., 91., 116., 121., 128., 85.},
        {173., 130., 101., 130., 121., 70., 151., 132.},
        {182., 91., 115., 25., 128., 151., 66., 92.},
        {125., 72., 142., 61., 85., 132., 92., 156.},
    };
    low_rank_4x4_ = Array2D<float>{
        // x = [[1, 2, 3, 4], [1, -1, 1, -1]]
        // matmul(x.T, x)
        {2, 1, 4, 3},
        {1, 5, 5, 9},
        {4, 5, 10, 11},
        {3, 9, 11, 17},
    };
  }
  void TearDown() override { ClientLibraryTestBase::TearDown(); }

  Array3D<float> get_unit_matrix_3d(const Array3D<float>& matrix) {
    Array3D<float> result(matrix.n1(), matrix.n2(), matrix.n3(), 0.0);
    for (int i = 0; i < matrix.n1(); ++i) {
      for (int j = 0; j < matrix.n2(); ++j) {
        result({i, j, j}) = 1.0;
      }
    }
    return result;
  }

  Array3D<float> ExtractTriangularMatrix(const Array3D<float>& matrix,
                                         bool lower) {
    Array3D<float> result(matrix);
    for (int i = 0; i < result.n1(); ++i) {
      for (int j = 0; j < result.n2(); ++j) {
        if (lower) {
          for (int k = j + 1; k < result.n3(); ++k) {
            result({i, j, k}) = 0.0;
          }
        } else {
          for (int k = 0; k < j; ++k) {
            result({i, j, k}) = 0.0;
          }
        }
      }
    }
    return result;
  }

  XlaOp ComputeMatmulVWVt(SelfAdjointEigenResult result, XlaBuilder* builder) {
    Shape shape = builder->GetShape(result.v).ValueOrDie();
    std::vector<int64> out_dims = shape.dimensions();
    std::vector<int64> broadcast_dims(shape.rank() - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);

    broadcast_dims[shape.rank() - 2] = shape.rank() - 1;
    auto vw = Mul(result.v, BroadcastInDim(result.w, out_dims, broadcast_dims));
    return BatchDot(vw, TransposeInMinorDims(result.v),
                    PrecisionConfig::HIGHEST);
  }

  XlaOp GetAverageAbsoluteError(XlaOp m1, XlaOp m2, XlaBuilder* builder) {
    Shape shape = builder->GetShape(m1).ValueOrDie();
    int64 size = 1;
    for (auto d : shape.dimensions()) {
      size *= d;
    }
    return ReduceAll(Abs(m1 - m2), ConstantR0WithType(builder, F32, 0),
                     CreateScalarAddComputation(F32, builder)) /
           ConstantR0WithType(builder, F32, size);
  }

  Array2D<float> GenerateRandomSymmetricMatrix(int size) {
    Array2D<float> result{size, size, 0.0};
    result.FillRandom(10 /* stddev */, 2 /* mean */);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < i; ++j) {
        result({j, i}) = result({i, j});
      }
    }
    return result;
  }

  Array3D<float> batch_3d_4x4_;
  Array2D<float> matrix2d_8x8_;
  Array2D<float> low_rank_4x4_;
  Array2D<int> wrong_type_4x4_;
};

XLA_TEST_F(SelfAdjointEigenTest, Test_VWVt_EQ_A_2x4x4) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x4_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  ComputeMatmulVWVt(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_VWVt_EQ_A_Lower_2x4x4) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(
      ExtractTriangularMatrix(batch_3d_4x4_, true), 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  ComputeMatmulVWVt(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_VWVt_EQ_A_Upper_2x4x4) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(
      ExtractTriangularMatrix(batch_3d_4x4_, false), 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a, false);
  ComputeMatmulVWVt(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_Orthogonality_2x4x4) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x4_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  BatchDot(result.v, TransposeInMinorDims(result.v), PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, get_unit_matrix_3d(batch_3d_4x4_),
                             {a_data.get()}, ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_VtWV_EQ_A_Rank_Deficient_4x4) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(low_rank_4x4_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  ComputeMatmulVWVt(result, &builder);

  ComputeAndCompareR2<float>(&builder, low_rank_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_Eigen_8x8) {
  XlaBuilder builder(TestName());

  // This is computed by numpy.linalg.eigh with float32.
  std::vector<float> expected{-182.69205, -116.86245, -105.74489, -9.545369,
                              37.81711,   104.732285, 120.29153,  868.00385};

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(matrix2d_8x8_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  Sort(result.w);

  ComputeAndCompareR1<float>(&builder, expected, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Test_Orthogonality_8x8) {
  XlaBuilder builder(TestName());

  float expected_vals = 1e-3;

  XlaOp a;
  auto a_data = CreateR2Parameter<float>(matrix2d_8x8_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  // np.sum(norm(eye(n) - matmul(conj(T(v)), v)) / n**2
  GetAverageAbsoluteError(IdentityMatrix(&builder, F32, 8, 8),
                          BatchDot(TransposeInMinorDims(result.v), result.v),
                          &builder);

  ComputeAndCompareR0<float>(&builder, expected_vals, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SelfAdjointEigenTest, Wrong_Type_Int) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR2Parameter<int>(wrong_type_4x4_, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  EXPECT_FALSE(result.v.valid());
  EXPECT_FALSE(result.w.valid());
}

XLA_TEST_F(SelfAdjointEigenTest, Various_Size_Random_Matrix_8x8) {
  XlaBuilder builder(TestName());
  int size = 8;
  Array2D<float> a_val = GenerateRandomSymmetricMatrix(size);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  GetAverageAbsoluteError(ComputeMatmulVWVt(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(SelfAdjointEigenTest, Various_Size_Random_Matrix_16x16) {
  XlaBuilder builder(TestName());
  int size = 16;
  Array2D<float> a_val = GenerateRandomSymmetricMatrix(size);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  GetAverageAbsoluteError(ComputeMatmulVWVt(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(SelfAdjointEigenTest, Various_Size_Random_Matrix_32x32) {
  XlaBuilder builder(TestName());
  int size = 32;
  Array2D<float> a_val = GenerateRandomSymmetricMatrix(size);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  GetAverageAbsoluteError(ComputeMatmulVWVt(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(SelfAdjointEigenTest, Various_Size_Random_Matrix_64x64) {
  XlaBuilder builder(TestName());
  int size = 64;
  Array2D<float> a_val = GenerateRandomSymmetricMatrix(size);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SelfAdjointEigen(a);
  GetAverageAbsoluteError(ComputeMatmulVWVt(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

}  // namespace xla
