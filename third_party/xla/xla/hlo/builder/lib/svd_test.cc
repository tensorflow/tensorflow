/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/svd.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include "absl/status/statusor.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {

class SVDTest : public ClientLibraryTestRunnerMixin<
                    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 protected:
  void SetUp() override {
    ClientLibraryTestRunnerMixin<
        HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>::SetUp();
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

  Array3D<float> GetUnitMatrix3D(int32_t batch_dim, int32_t mat_dim) {
    Array3D<float> result(batch_dim, mat_dim, mat_dim, 0.0);
    for (int i = 0; i < batch_dim; ++i) {
      for (int j = 0; j < mat_dim; ++j) {
        result({i, j, j}) = 1.0;
      }
    }
    return result;
  }

  XlaOp ComputeMatmulUDVT(SVDResult result, XlaBuilder* builder) {
    Shape u_shape = builder->GetShape(result.u).value();
    Shape v_shape = builder->GetShape(result.v).value();

    int64_t m = ShapeUtil::GetDimension(u_shape, -1);
    int64_t n = ShapeUtil::GetDimension(v_shape, -1);

    auto v = result.v;
    auto u = result.u;
    auto d = result.d;

    if (m > n) {
      u = SliceInMinorDims(u, {0, 0}, {m, n});
    } else if (m < n) {
      v = SliceInMinorDims(v, {0, 0}, {n, m});
    }

    int num_dims = u_shape.dimensions().size();
    std::vector<int64_t> broadcast_dims(num_dims - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    broadcast_dims[num_dims - 2] = num_dims - 1;
    return BatchDot(Mul(u, d, broadcast_dims), TransposeInMinorDims(v),
                    PrecisionConfig::HIGHEST);
  }

  XlaOp GetAverageAbsoluteError(XlaOp m1, XlaOp m2, XlaBuilder* builder) {
    Shape shape = builder->GetShape(m1).value();
    int64_t size = 1;
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

TEST_F(SVDTest, Simple2D) {
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

  ComputeAndCompareR2<float>(&builder, simple_2d_4x4_, {&a_data},
                             ErrorSpec(1e-3, 1e-3));
}

TEST_F(SVDTest, Test_VWVt_EQ_A_2x4x5) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);

  ComputeAndCompareR3<float>(&builder, batch_3d_4x5_, {&a_data},
                             ErrorSpec(1e-3, 1e-3));
}

TEST_F(SVDTest, Test_Orthogonality_U) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  ComputeMatmulUDVT(result, &builder);
  BatchDot(result.u, TransposeInMinorDims(result.u));

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 4), {&a_data},
                             ErrorSpec(1e-2, 1e-2));
}

TEST_F(SVDTest, Test_Orthogonality_V) {
  XlaBuilder builder(TestName());

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  BatchDot(result.v, TransposeInMinorDims(result.v), PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, GetUnitMatrix3D(2, 5), {&a_data},
                             ErrorSpec(1e-3, 1e-3));
}

TEST_F(SVDTest, TestSingleValuesMatchNumpy) {
  XlaBuilder builder(TestName());

  auto singular_values = Array2D<float>{
      {431.05153007, 49.88334164, 20.94464584, 3.24845468},
      {179.73128591, 68.05162245, 21.77679503, 13.94319712},
  };

  XlaOp a;
  auto a_data = CreateR3Parameter<float>(batch_3d_4x5_, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-8);
  Add(result.d, ZerosLike(result.d));

  ComputeAndCompareR2<float>(&builder, singular_values, {&a_data},
                             ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
TEST_F(SVDTest, Various_Size_Random_Matrix_512x128) {
  if (test::DeviceIs(test::kInterpreter)) {
    GTEST_SKIP();
  }
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(SVDTest, Various_Size_Random_Matrix_128x256) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

TEST_F(SVDTest, Various_Size_Random_Matrix_256x128) {
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(256, 128);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter backend.
TEST_F(SVDTest, Various_Size_Random_Matrix_128x512) {
  if (test::DeviceIs(test::kInterpreter)) {
    GTEST_SKIP();
  }
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(128, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

// Too slow on the interpreter and CPU backends.
TEST_F(SVDTest, Various_Size_Random_Matrix_512x256) {
  if (test::DeviceIsOneOf({test::kCpu, test::kInterpreter})) {
    GTEST_SKIP();
  }
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 256);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

// Too slow on the CPU, GPU and interpreter backends.
TEST_F(SVDTest, Various_Size_Random_Matrix_512x512) {
  if (test::DeviceTypeIsOneOf({test::kCpu, test::kGpu, test::kInterpreter})) {
    GTEST_SKIP();
  }
  XlaBuilder builder(TestName());
  Array2D<float> a_val = GenerateRandomMatrix(512, 512);
  XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_val, 0, "a", &builder, &a);
  auto result = SVD(a, 100, 1e-4);
  GetAverageAbsoluteError(ComputeMatmulUDVT(result, &builder), a, &builder);

  ComputeAndCompareR0<float>(&builder, 1e-3, {&a_data}, ErrorSpec(1e-3, 1e-3));
}

}  // namespace xla
