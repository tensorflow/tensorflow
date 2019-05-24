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

#include "tensorflow/compiler/xla/client/lib/svd.h"

#include <utility>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {

class SVDTest : public ClientLibraryTestBase {
 protected:
  void SetUp() override {
    ClientLibraryTestBase::SetUp();
    batch_3d_4x5_ = Array3D<float>{
        {
            {4, 6, 8, 10, 1},
            {6, 45, 54, 63, 1},
            {8, 54, 146, 166, 1},
            {10, 63, 166, 310, 1},
        },
        {
            {16, 24, 8, 12, 6},
            {24, 61, 82, 48, 5},
            {8, 82, 100, 6, 4},
            {12, 48, 6, 62, 3},
        },
    };
  }
  void TearDown() override { ClientLibraryTestBase::TearDown(); }

  Array3D<float> GetUnitMatrix3D(int32 batch_dim, int32 mat_dim) {
    Array3D<float> result(batch_dim, mat_dim, mat_dim, 0.0);
    for (int i = 0; i < batch_dim; ++i) {
      for (int j = 0; j < mat_dim; ++j) {
        result({i, j, j}) = 1.0;
      }
    }
    return result;
  }

  XlaOp ComputeMatmulUDVT(SVDResult result, XlaBuilder* builder) {
    Shape u_shape = builder->GetShape(result.u).ValueOrDie();
    Shape v_shape = builder->GetShape(result.v).ValueOrDie();

    int64 m = ShapeUtil::GetDimension(u_shape, -1);
    int64 n = ShapeUtil::GetDimension(v_shape, -1);

    auto v = result.v;
    auto u = result.u;
    auto d = result.d;

    if (m > n) {
      u = SliceInMinorDims(u, {0, 0}, {m, n});
    } else if (m < n) {
      v = SliceInMinorDims(v, {0, 0}, {n, m});
    }

    int num_dims = u_shape.rank();
    std::vector<int64> broadcast_dims(num_dims - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    broadcast_dims[num_dims - 2] = num_dims - 1;
    return BatchDot(Mul(u, d, broadcast_dims), TransposeInMinorDims(v),
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

  Array2D<float> GenerateRandomMatrix(int xsize, int ysize) {
    Array2D<float> result{xsize, ysize, 0.0};
    result.FillRandom(10 /* stddev */, 2 /* mean */);
    return result;
  }

  Array3D<float> batch_3d_4x5_;
};

XLA_TEST_F(SVDTest, Simple2D) {
  XlaBuilder builder(TestName());

  Array2D<float> simple_2d_4x4_ = Array2D<float>{
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  };
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(simple_2d_4x4_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-6);
  ComputeMatmulUDVT(result, &builder);

  ComputeAndCompareR2<float>(&builder, simple_2d_4x4_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Test_VWVt_EQ_A_2x4x5) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x5_, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Test_Orthogonality_U) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);
  BatchDot(result.u, TransposeInMinorDims(result.u));

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 4), {a_data.get()},
                             ErrorSpec(1e-2, 1e-2));
}

XLA_TEST_F(SVDTest, Test_Orthogonality_V) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  BatchDot(result.v, TransposeInMinorDims(result.v), PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 5), {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, TestSingleValuesMatchNumpy) {
  XlaBuilder builder(TestName());

  auto singular_values = Array2D<float>{
      {431.05153007, 49.88334164, 20.94464584, 3.24845468},
      {179.73128591, 68.05162245, 21.77679503, 13.94319712},
  };

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  Add(result.d, ZerosLike(result.d));

  ComputeAndCompareR2<float>(&builder, singular_values, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
XLA_TEST_F(SVDTest,
           DISABLED_ON_INTERPRETER(Various_Size_Random_Matrix_512x128)) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Various_Size_Random_Matrix_128x256) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

XLA_TEST_F(SVDTest, Various_Size_Random_Matrix_256x128) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(256, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
XLA_TEST_F(SVDTest,
           DISABLED_ON_INTERPRETER(Various_Size_Random_Matrix_128x512)) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter and CPU backends.
XLA_TEST_F(SVDTest, DISABLED_ON_CPU(DISABLED_ON_INTERPRETER(
                        Various_Size_Random_Matrix_512x256))) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the CPU, GPU and interpreter backends.
XLA_TEST_F(SVDTest, DISABLED_ON_GPU(DISABLED_ON_CPU(DISABLED_ON_INTERPRETER(
                        Various_Size_Random_Matrix_512x512)))) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {a_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

}  // namespace xla
