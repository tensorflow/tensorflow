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

#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class BatchNormalizationTest : public ClientLibraryTestBase {
 protected:
  BatchNormalizationTest() : input_array_(kSamples, kZ, kY, kX) {
    Array2D<float> pz({
        // z0 z1
        {-1.0f, 4.1f},  // p0
        {2.0f, 4.1f},   // p1
        {5.0f, 4.4f},   // p2
    });
    input_array_.FillWithPZ(pz);
    input_literal_ = *LiteralUtil::CreateR4FromArray4D(input_array_);
    CHECK_EQ(kSamples, input_array_.planes());
    CHECK_EQ(kZ, input_array_.depth());
    CHECK_EQ(kY, input_array_.height());
    CHECK_EQ(kY, input_array_.width());
  }

  static constexpr int64 kSamples = 3;
  static constexpr int64 kX = 1;
  static constexpr int64 kY = 1;
  static constexpr int64 kZ = 2;

  Array4D<float> input_array_;
  Literal input_literal_;
  const ErrorSpec error_spec_{0.001, 0.001};
};

TEST_F(BatchNormalizationTest, SubtractInZ) {
  ComputationBuilder builder(client_, "subtract_in_z_one_sample");
  auto x = builder.ConstantLiteral(input_literal_);
  auto y = builder.ConstantR1<float>({3.14, 4.25});
  builder.Sub(x, y, /*broadcast_dimensions=*/{1});

  Array4D<float> expected(kSamples, kZ, kY, kX);
  Array2D<float> pz({
      {-1.0f - 3.14f, 4.1f - 4.25f},  // p0
      {2.0f - 3.14f, 4.1f - 4.25f},   // p1
      {5.0f - 3.14f, 4.4f - 4.25f},   // p2
  });
  expected.FillWithPZ(pz);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

TEST_F(BatchNormalizationTest, SquareTesseractElementwise) {
  ComputationBuilder builder(client_, "square_tesseract_elementwise");
  auto x = builder.ConstantLiteral(input_literal_);
  builder.SquareF32(x);

  Array4D<float> expected(kSamples, kZ, kY, kX);
  Array2D<float> expected_pz({
      {std::pow(-1.0f, 2.0f), std::pow(4.1f, 2.0f)},
      {std::pow(2.0f, 2.0f), std::pow(4.1f, 2.0f)},
      {std::pow(5.0f, 2.0f), std::pow(4.4f, 2.0f)},
  });
  expected.FillWithPZ(expected_pz);
  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

TEST_F(BatchNormalizationTest, SumToZ) {
  ComputationBuilder builder(client_, "sum_to_z");
  auto input_activations = builder.ConstantLiteral(input_literal_);
  Computation add = CreateScalarAddComputation(F32, &builder);
  // Reduce all but the Z dimension.
  builder.Reduce(input_activations, builder.ConstantR0<float>(0.0f), add,
                 {0, 2, 3});

  std::vector<float> expected = {6, 12.6};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

TEST_F(BatchNormalizationTest, SquareAndReduce) {
  ComputationBuilder builder(client_, "square_and_reduce");
  auto input_activations = builder.ConstantLiteral(input_literal_);
  auto set_means = builder.ConstantR1<float>({2.f, 4.2f});
  auto activation_deviations = builder.Sub(input_activations, set_means,
                                           /*broadcast_dimensions=*/{1});
  Computation add = CreateScalarAddComputation(F32, &builder);
  auto dev_squares = builder.SquareF32(activation_deviations);
  auto sum_of_squares = builder.Reduce(
      dev_squares, builder.ConstantR0<float>(0.0f), add, {0, 2, 3});

  std::vector<float> expected = {18, 0.06};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

TEST_F(BatchNormalizationTest, VarianceToStddev) {
  ComputationBuilder builder(client_, "variance_to_stddev");
  auto variance = builder.ConstantR1<float>({6.f, .02f});
  auto sqrt = builder.SqrtF32(variance);

  std::vector<float> expected = {2.44948974f, 0.14142136f};
  ComputeAndCompareR1<float>(&builder, expected, {}, error_spec_);
}

// Compare against a forward batch normalization example in the NN spec
// reference.
TEST_F(BatchNormalizationTest, SpecComparisonForward) {
  ComputationBuilder builder(client_, "batch_normalize_per_spec");
  auto input_activations =
      builder.CheckShape(builder.ConstantLiteral(input_literal_),
                         ShapeUtil::MakeShape(F32, {3, 2, 1, 1}));
  auto gamma = builder.ConstantR1<float>({1.0, 1.0});
  auto beta = builder.ConstantR1<float>({0.0, 0.0});
  Computation add = CreateScalarAddComputation(F32, &builder);
  // Reduce all dimensions except dimension 1.
  Shape TwoElementVectorF32 = ShapeUtil::MakeShape(F32, {2});
  auto sum = builder.CheckShape(
      builder.Reduce(input_activations, builder.ConstantR0<float>(0.0f), add,
                     /*dimensions_to_reduce=*/{0, 2, 3}),
      TwoElementVectorF32);
  auto input_shape = builder.GetShape(input_activations).ConsumeValueOrDie();
  auto sum_shape = builder.GetShape(sum).ConsumeValueOrDie();
  auto count = builder.ConstantR0<float>(ShapeUtil::ElementsIn(*input_shape) /
                                         ShapeUtil::ElementsIn(*sum_shape));
  auto set_means = builder.Div(sum, count);

  const float kEpsilon = 1e-9f;
  auto epsilon = builder.ConstantR0<float>(kEpsilon);
  auto epsilon2 = builder.ConstantR1<float>({kEpsilon, kEpsilon});
  auto activation_deviations = builder.Sub(input_activations, set_means,
                                           /*broadcast_dimensions=*/{1});
  auto dev_squares = builder.SquareF32(activation_deviations);
  auto sum_of_squares = builder.CheckShape(
      builder.Reduce(dev_squares, builder.ConstantR0<float>(0.0f), add,
                     /*dimensions_to_reduce=*/{0, 2, 3}),
      TwoElementVectorF32);
  auto variance = builder.Div(sum_of_squares, count);
  auto standard_deviation = builder.SqrtF32(variance);
  auto standard_deviation_above_epsilon = builder.CheckShape(
      builder.Gt(standard_deviation, epsilon), ShapeUtil::MakeShape(PRED, {2}));
  auto gt_eps = builder.Select(standard_deviation_above_epsilon,
                               standard_deviation, epsilon2);
  auto normalization_factors = builder.ReciprocalF32(gt_eps);
  auto normalized_input_activations =
      builder.Mul(activation_deviations, normalization_factors,
                  /*broadcast_dimensions=*/{1});
  /* auto output_activations = */ builder.Add(
      builder.Mul(normalized_input_activations, gamma,
                  /*broadcast_dimensions=*/{1}),
      beta, /*broadcast_dimensions=*/{1});

  Array4D<float> expected(kSamples, kZ, kY, kX);
  Array2D<float> pz({
      {-3.f / std::sqrt(6.f), -.1f / std::sqrt(.02f)},
      {0.f, -.1f / std::sqrt(.02f)},
      {3.f / std::sqrt(6.f), .2f / std::sqrt(.02f)},
  });
  expected.FillWithPZ(pz);

  ComputeAndCompareR4<float>(&builder, expected, {}, error_spec_);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
