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
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
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
    input_literal_ = *Literal::CreateR4FromArray4D(input_array_);
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

struct BatchNormTestParam {
  std::vector<int64> bounds;
  int64 feature_index;
  float random_value_mean;
  float random_value_var;
};

// Tests to test the fused operation of BatchNorm.
class BatchNormTest : public ClientLibraryTestBase,
                      public ::testing::WithParamInterface<BatchNormTestParam> {
};

XLA_TEST_P(BatchNormTest, RandomizedTests) {
  float epsilon = 0.001;
  ComputationBuilder builder(client_, TestName());
  const std::vector<int64>& bounds = GetParam().bounds;
  Array4D<float> input_array(bounds[0], bounds[1], bounds[2], bounds[3]);
  input_array.FillRandom(GetParam().random_value_var,
                         GetParam().random_value_mean);

  const int64 feature_index = GetParam().feature_index;
  const int64 num_elements_per_feature =
      Product(bounds) / bounds[feature_index];
  const int64 feature_bound = bounds[feature_index];
  std::vector<float> offset(feature_bound, 1);
  std::vector<float> scale(feature_bound, 2);

  auto input_squared =
      ReferenceUtil::MapArray4D(input_array, [](float a) { return a * a; });
  std::vector<int64> reduce_dims;
  for (int64 i = 0; i < static_cast<int64>(bounds.size()); ++i) {
    if (i != feature_index) {
      reduce_dims.push_back(i);
    }
  }

  auto sum =
      ReferenceUtil::Reduce4DTo1D(input_array, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  auto sum_squared =
      ReferenceUtil::Reduce4DTo1D(*input_squared, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  std::vector<float> mean(feature_bound);

  for (int64 i = 0; i < feature_bound; ++i) {
    mean[i] = sum[i] / num_elements_per_feature;
  }

  std::vector<float> mean_square(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    mean_square[i] = mean[i] * mean[i];
  }

  std::vector<float> square_mean(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    square_mean[i] = sum_squared[i] / num_elements_per_feature;
  }

  std::vector<float> var(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    var[i] = square_mean[i] - mean_square[i];
  }

  Array4D<float> mean4D =
      *ReferenceUtil::Broadcast1DTo4D(mean, bounds, feature_index);
  auto var4D = *ReferenceUtil::Broadcast1DTo4D(var, bounds, feature_index);
  auto scale4D = *ReferenceUtil::Broadcast1DTo4D(scale, bounds, feature_index);
  auto offset4D =
      *ReferenceUtil::Broadcast1DTo4D(offset, bounds, feature_index);

  auto normalized = *ReferenceUtil::BatchNorm4D(input_array, mean4D, var4D,
                                                scale4D, offset4D, epsilon);

  auto expected_normalized = Literal::CreateR4FromArray4D<float>(normalized);

  auto offset_literal = Literal::CreateR1<float>(offset);
  auto scale_literal = Literal::CreateR1<float>(scale);
  auto input_literal = Literal::CreateR4FromArray4D<float>(input_array);

  auto input_activations =
      builder.Parameter(0, input_literal->shape(), "input");
  auto scale_activations =
      builder.Parameter(1, scale_literal->shape(), "offset");
  auto offset_activations =
      builder.Parameter(2, offset_literal->shape(), "scale");

  auto expected = *Literal::MakeTuple({expected_normalized.get(),
                                       Literal::CreateR1<float>(mean).get(),
                                       Literal::CreateR1<float>(var).get()});

  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> scale_data =
      client_->TransferToServer(*scale_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> offset_data =
      client_->TransferToServer(*offset_literal).ConsumeValueOrDie();

  builder.BatchNormTraining(input_activations, scale_activations,
                            offset_activations, epsilon, feature_index);

  ComputeAndCompareTuple(
      &builder, expected,
      {input_data.get(), scale_data.get(), offset_data.get()},
      ErrorSpec(0.01, 1));
}

XLA_TEST_P(BatchNormTest, RandomizedInferencingTests) {
  float epsilon = 0.001;
  ComputationBuilder builder(client_, TestName());
  const std::vector<int64>& bounds = GetParam().bounds;
  Array4D<float> input_array(bounds[0], bounds[1], bounds[2], bounds[3]);
  input_array.FillRandom(GetParam().random_value_var,
                         GetParam().random_value_mean);

  const int64 feature_index = GetParam().feature_index;
  const int64 num_elements_per_feature =
      Product(bounds) / bounds[feature_index];
  const int64 feature_bound = bounds[feature_index];
  std::vector<float> offset(feature_bound, 1);
  std::vector<float> scale(feature_bound, 2);

  auto input_squared =
      ReferenceUtil::MapArray4D(input_array, [](float a) { return a * a; });
  std::vector<int64> reduce_dims;
  for (int64 i = 0; i < static_cast<int64>(bounds.size()); ++i) {
    if (i != feature_index) {
      reduce_dims.push_back(i);
    }
  }

  auto sum =
      ReferenceUtil::Reduce4DTo1D(input_array, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  auto sum_squared =
      ReferenceUtil::Reduce4DTo1D(*input_squared, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  std::vector<float> mean(feature_bound);

  for (int64 i = 0; i < feature_bound; ++i) {
    mean[i] = sum[i] / num_elements_per_feature;
  }

  std::vector<float> mean_square(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    mean_square[i] = mean[i] * mean[i];
  }

  std::vector<float> square_mean(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    square_mean[i] = sum_squared[i] / num_elements_per_feature;
  }

  std::vector<float> var(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    var[i] = square_mean[i] - mean_square[i];
  }

  Array4D<float> mean4D =
      *ReferenceUtil::Broadcast1DTo4D(mean, bounds, feature_index);
  auto var4D = *ReferenceUtil::Broadcast1DTo4D(var, bounds, feature_index);
  auto scale4D = *ReferenceUtil::Broadcast1DTo4D(scale, bounds, feature_index);
  auto offset4D =
      *ReferenceUtil::Broadcast1DTo4D(offset, bounds, feature_index);

  auto normalized = *ReferenceUtil::BatchNorm4D(input_array, mean4D, var4D,
                                                scale4D, offset4D, epsilon);

  auto offset_literal = Literal::CreateR1<float>(offset);
  auto scale_literal = Literal::CreateR1<float>(scale);
  auto mean_literal = Literal::CreateR1<float>(mean);
  auto var_literal = Literal::CreateR1<float>(var);
  auto input_literal = Literal::CreateR4FromArray4D<float>(input_array);

  auto input_activations =
      builder.Parameter(0, input_literal->shape(), "input");
  auto scale_activations =
      builder.Parameter(1, scale_literal->shape(), "offset");
  auto offset_activations =
      builder.Parameter(2, offset_literal->shape(), "scale");
  auto mean_activations = builder.Parameter(3, mean_literal->shape(), "mean");
  auto variance_activations =
      builder.Parameter(4, var_literal->shape(), "variance");

  Array4D<float> expected = normalized;

  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> scale_data =
      client_->TransferToServer(*scale_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> offset_data =
      client_->TransferToServer(*offset_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> mean_data =
      client_->TransferToServer(*mean_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> variance_data =
      client_->TransferToServer(*var_literal).ConsumeValueOrDie();

  builder.BatchNormInference(input_activations, scale_activations,
                             offset_activations, mean_activations,
                             variance_activations, epsilon, feature_index);

  ComputeAndCompareR4<float>(
      &builder, expected,
      {input_data.get(), scale_data.get(), offset_data.get(), mean_data.get(),
       variance_data.get()},
      ErrorSpec(0.01, 1));
}

XLA_TEST_P(BatchNormTest, RandomizedGradTests) {
  float epsilon = 0.001;
  ComputationBuilder builder(client_, TestName());
  const std::vector<int64>& bounds = GetParam().bounds;
  Array4D<float> input_array(bounds[0], bounds[1], bounds[2], bounds[3]);
  input_array.FillRandom(GetParam().random_value_var,
                         GetParam().random_value_mean);

  Array4D<float> grad_output_array(bounds[0], bounds[1], bounds[2], bounds[3]);
  grad_output_array.FillRandom(GetParam().random_value_var,
                               GetParam().random_value_mean);

  const int64 feature_index = GetParam().feature_index;
  const int64 num_elements_per_feature =
      Product(bounds) / bounds[feature_index];
  const int64 feature_bound = bounds[feature_index];
  std::vector<float> scale(feature_bound, 2);

  auto input_squared =
      ReferenceUtil::MapArray4D(input_array, [](float a) { return a * a; });
  std::vector<int64> reduce_dims;
  for (int64 i = 0; i < static_cast<int64>(bounds.size()); ++i) {
    if (i != feature_index) {
      reduce_dims.push_back(i);
    }
  }

  auto sum =
      ReferenceUtil::Reduce4DTo1D(input_array, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  auto sum_squared =
      ReferenceUtil::Reduce4DTo1D(*input_squared, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  std::vector<float> mean(feature_bound);

  for (int64 i = 0; i < feature_bound; ++i) {
    mean[i] = sum[i] / num_elements_per_feature;
  }

  std::vector<float> mean_square(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    mean_square[i] = mean[i] * mean[i];
  }

  std::vector<float> square_mean(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    square_mean[i] = sum_squared[i] / num_elements_per_feature;
  }

  std::vector<float> var(feature_bound);
  for (int64 i = 0; i < feature_bound; ++i) {
    var[i] = square_mean[i] - mean_square[i];
  }

  Array4D<float> mean4D =
      *ReferenceUtil::Broadcast1DTo4D(mean, bounds, feature_index);
  auto var4D = *ReferenceUtil::Broadcast1DTo4D(var, bounds, feature_index);
  auto scale4D = *ReferenceUtil::Broadcast1DTo4D(scale, bounds, feature_index);

  auto var_add_epsilon = *ReferenceUtil::MapArray4D(
      var4D, [epsilon](float a) { return a + epsilon; });

  auto rsqrt_var_add_epsilon = *ReferenceUtil::MapArray4D(
      var_add_epsilon, [epsilon](float a) { return 1 / std::sqrt(a); });

  auto grad_output_times_var =
      *ReferenceUtil::MapArray4D(grad_output_array, var_add_epsilon,
                                 [](float a, float b) { return a * b; });

  auto activation_shifted = *ReferenceUtil::MapArray4D(
      input_array, mean4D, [](float a, float b) { return a - b; });

  auto activation_shifted_times_grad_output =
      *ReferenceUtil::MapArray4D(grad_output_array, activation_shifted,
                                 [](float a, float b) { return a * b; });

  auto grad_scale_before_reduction = *ReferenceUtil::MapArray4D(
      activation_shifted_times_grad_output, rsqrt_var_add_epsilon,
      [](float a, float b) { return a * b; });

  auto grad_scale = ReferenceUtil::Reduce4DTo1D(
      grad_scale_before_reduction, /*init=*/0.0f, reduce_dims,
      [](float a, float b) { return a + b; });

  auto grad_offset =
      ReferenceUtil::Reduce4DTo1D(grad_output_array, /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; });

  auto scale_times_rsqrt_var_add_epsilon = *ReferenceUtil::MapArray4D(
      scale4D, rsqrt_var_add_epsilon, [](float a, float b) { return a * b; });

  auto I1 = *ReferenceUtil::MapArray4D(
      grad_output_array, [&](float a) { return num_elements_per_feature * a; });

  auto I2 = *ReferenceUtil::Broadcast1DTo4D(grad_offset, bounds, feature_index);

  // I3 = sum(output_grad * (activation - mean(activation)))
  auto I3 = *ReferenceUtil::Broadcast1DTo4D(
      ReferenceUtil::Reduce4DTo1D(activation_shifted_times_grad_output,
                                  /*init=*/0.0f, reduce_dims,
                                  [](float a, float b) { return a + b; }),
      bounds, feature_index);

  // I4 = (activation - mean(activation)) *
  //   sum(output_grad * (activation - mean(activation)))
  auto I4 = *ReferenceUtil::MapArray4D(I3, activation_shifted,
                                       [](float a, float b) { return a * b; });

  // I5 = (activation - mean(activation)) *
  //   sum(output_grad * (activation - mean(activation))) / (variance +
  //   epsilon))
  auto I5 = *ReferenceUtil::MapArray4D(I4, var_add_epsilon,
                                       [](float a, float b) { return a / b; });

  auto grad_activation = *ReferenceUtil::MapArray4D(
      I1, I2, [](float a, float b) { return a - b; });

  grad_activation = *ReferenceUtil::MapArray4D(
      grad_activation, I5, [](float a, float b) { return a - b; });

  grad_activation = *ReferenceUtil::MapArray4D(
      grad_activation, scale4D, [](float a, float b) { return a * b; });

  grad_activation = *ReferenceUtil::MapArray4D(
      grad_activation, rsqrt_var_add_epsilon,
      [=](float a, float b) { return a * b / num_elements_per_feature; });

  auto expected_grad_activation =
      Literal::CreateR4FromArray4D<float>(grad_activation);

  auto input_literal = Literal::CreateR4FromArray4D<float>(input_array);
  auto scale_literal = Literal::CreateR1<float>(scale);
  auto mean_literal = Literal::CreateR1<float>(mean);
  auto var_literal = Literal::CreateR1<float>(var);
  auto grad_output_literal =
      Literal::CreateR4FromArray4D<float>(grad_output_array);

  auto input_parameter = builder.Parameter(0, input_literal->shape(), "input");
  auto scale_parameter = builder.Parameter(1, scale_literal->shape(), "scale");
  auto mean_parameter = builder.Parameter(2, mean_literal->shape(), "mean");
  auto var_parameter = builder.Parameter(3, var_literal->shape(), "variance");
  auto grad_output_parameter =
      builder.Parameter(4, grad_output_literal->shape(), "grad_output");

  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> scale_data =
      client_->TransferToServer(*scale_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> mean_data =
      client_->TransferToServer(*mean_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> var_data =
      client_->TransferToServer(*var_literal).ConsumeValueOrDie();
  std::unique_ptr<GlobalData> grad_output_data =
      client_->TransferToServer(*grad_output_literal).ConsumeValueOrDie();

  auto t = builder.BatchNormGrad(input_parameter, scale_parameter,
                                 mean_parameter, var_parameter,
                                 grad_output_parameter, epsilon, feature_index);

  auto expected =
      *Literal::MakeTuple({expected_grad_activation.get(),
                           Literal::CreateR1<float>(grad_scale).get(),
                           Literal::CreateR1<float>(grad_offset).get()});

  ComputeAndCompareTuple(&builder, expected,
                         {input_data.get(), scale_data.get(), mean_data.get(),
                          var_data.get(), grad_output_data.get()},
                         ErrorSpec(0.01, 1));
}

INSTANTIATE_TEST_CASE_P(
    BatchNormTest_Instantiation, BatchNormTest,
    ::testing::Values(BatchNormTestParam{{2, 2, 2, 2}, 0, 100.2f, 200.0f},
                      BatchNormTestParam{{2, 2, 2, 2}, 3, 300.f, 400.0f},

                      BatchNormTestParam{{1, 10, 1, 1}, 0, 10.1f, 20.1f},
                      BatchNormTestParam{{10, 10, 10, 10}, 1, 3.14f, 314.15f},
                      BatchNormTestParam{{10, 10, 10, 10}, 2, 666.6f, 777.7f},
                      BatchNormTestParam{{10, 10, 10, 10}, 1, -666.6f, 777.7f},
                      BatchNormTestParam{{10, 10, 10, 10}, 2, 0.f, 777.7f},
                      BatchNormTestParam{{1, 1, 10, 130}, 2, 0.f, 777.7f},
                      BatchNormTestParam{{1, 1, 130, 11}, 2, 0.f, 777.7f},
                      BatchNormTestParam{{1, 1, 10, 1}, 3, 888.8f, 9.9f},

                      BatchNormTestParam{{24, 129, 1, 2}, 2, 10000, 10000},
                      BatchNormTestParam{{24, 129, 1, 2}, 3, 10000, 10000},

                      // Feature on low dimension to trigger relayout, test
                      // internal logical to physical dimension calculation
                      // is correct after relayout.
                      BatchNormTestParam{{1, 2, 3, 4}, 0, 100, 100}));

XLA_TEST_F(BatchNormTest, BasicTraining) {
  const int kFeatureIndex = 3;
  ComputationBuilder builder(client_, TestName());

  auto operand = builder.ConstantR4FromArray4D<float>(
      {{{{1.f, 2.f}}, {{3.f, 4.f}}}, {{{5.f, 6.f}}, {{7.f, 8.f}}}});

  auto scale = builder.ConstantR1<float>({2.0f, 3.0f});

  auto offset = builder.ConstantR1<float>({1.0f, 2.0f});

  auto tuple = builder.BatchNormTraining(operand, scale, offset,
                                         /*epsilon=*/0.001, kFeatureIndex);

  auto expected = *Literal::MakeTuple(
      {Literal::CreateR4<float>({{{{-1.6f, -2.0f}}, {{0.1f, 0.6f}}},
                                 {{{1.9f, 3.3f}}, {{3.7f, 6.0f}}}})
           .get(),
       Literal::CreateR1<float>({4, 5}).get(),
       Literal::CreateR1<float>({5, 5}).get()});

  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.1));
}

XLA_TEST_F(BatchNormTest, BasicTrainingOnSublane) {
  const int kFeatureIndex = 2;
  ComputationBuilder builder(client_, TestName());

  auto operand = builder.ConstantR4FromArray4D<float>(
      {{{{1.f}, {2.f}}, {{3.f}, {4.f}}}, {{{5.f}, {6.f}}, {{7.f}, {8.f}}}});

  auto scale = builder.ConstantR1<float>({2.0f, 3.0f});

  auto offset = builder.ConstantR1<float>({1.0f, 2.0f});

  auto tuple = builder.BatchNormTraining(operand, scale, offset,
                                         /*epsilon=*/0.001, kFeatureIndex);

  auto expected = *Literal::MakeTuple(
      {Literal::CreateR4<float>({{{{-1.6f}, {-2.0f}}, {{0.1f}, {0.6f}}},
                                 {{{1.9f}, {3.3f}}, {{3.7f}, {6.0f}}}})
           .get(),
       Literal::CreateR1<float>({4, 5}).get(),
       Literal::CreateR1<float>({5, 5}).get()});

  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.1));
}

XLA_TEST_F(BatchNormTest, DISABLED_ON_GPU(TrainingWithFeatureOnLowDimension)) {
  // Use 0 dimension as feature, tests layout analyzer.
  const int kFeatureIndex = 0;
  ComputationBuilder builder(client_, TestName());

  ComputationDataHandle h0;
  auto operand = CreateR3Parameter<float>(Array3D<float>(260, 2, 2, 1.0f),
                                          /*parameter_number=*/0, "operand",
                                          &builder, &h0);
  ComputationDataHandle h1;
  auto scale =
      CreateR1Parameter<float>(std::vector<float>(260, 1.0f),
                               /*parameter_number=*/1, "scale", &builder, &h1);
  ComputationDataHandle h2;
  auto offset =
      CreateR1Parameter<float>(std::vector<float>(260, 1.0f),
                               /*parameter_number=*/2, "offset", &builder, &h2);

  auto tuple = builder.BatchNormTraining(h0, h1, h2,
                                         /*epsilon=*/1, kFeatureIndex);

  auto expected = *Literal::MakeTuple(
      {Literal::CreateR3FromArray3D<float>(Array3D<float>(260, 2, 2, 1.0f))
           .get(),
       Literal::CreateR1<float>(std::vector<float>(260, 1.0f)).get(),
       Literal::CreateR1<float>(std::vector<float>(260, 0.0f)).get()});

  ComputeAndCompareTuple(&builder, expected,
                         {operand.get(), scale.get(), offset.get()},
                         ErrorSpec(0.1));
}

XLA_TEST_F(BatchNormTest, LargeEpsilonTest) {
  // Test the correctness of choosing a large epsilon value.
  const int kFeatureIndex = 2;
  ComputationBuilder builder(client_, TestName());

  ComputationDataHandle h0;
  auto operand = CreateR3Parameter<float>({{{0.0f}, {10.0f}, {20.0f}, {30.0f}}},
                                          /*parameter_number=*/0, "operand",
                                          &builder, &h0);
  ComputationDataHandle h1;
  auto scale =
      CreateR1Parameter<float>(std::vector<float>(1, 1.0f),
                               /*parameter_number=*/1, "scale", &builder, &h1);
  ComputationDataHandle h2;
  auto offset =
      CreateR1Parameter<float>(std::vector<float>(1, 0.0f),
                               /*parameter_number=*/2, "offset", &builder, &h2);

  // var = 125, mean = 15, epsilon = -100
  auto tuple = builder.BatchNormTraining(h0, h1, h2,
                                         /*epsilon=*/-100, kFeatureIndex);

  auto expected = *Literal::MakeTuple(
      {Literal::CreateR3FromArray3D<float>({{{-3.0f}, {-1.0f}, {1.0f}, {3.0f}}})
           .get(),
       Literal::CreateR1<float>(std::vector<float>(1, 15.0f)).get(),
       Literal::CreateR1<float>(std::vector<float>(1, 125.0f)).get()});

  ComputeAndCompareTuple(&builder, expected,
                         {operand.get(), scale.get(), offset.get()},
                         ErrorSpec(0.1));
}

XLA_TEST_F(BatchNormTest, BatchNormGradBasic) {
  const int kFeatureIndex = 2;
  ComputationBuilder builder(client_, TestName());

  auto operand =
      builder.ConstantR4FromArray4D<float>(Array4D<float>(2, 2, 2, 1, 0.0f));

  auto scale = builder.ConstantR1<float>({1.0f, 1.0f});

  auto mean = builder.ConstantR1<float>({0.0f, 0.0f});

  auto var = builder.ConstantR1<float>({1.0f, 1.0f});

  auto grad_output = builder.ConstantR4FromArray4D<float>(
      {{{{1.f}, {2.f}}, {{3.f}, {4.f}}}, {{{5.f}, {6.f}}, {{7.f}, {8.f}}}});

  builder.BatchNormGrad(operand, scale, mean, var, grad_output,
                        /*epsilon=*/0.0, kFeatureIndex);

  auto expected = *Literal::MakeTuple(
      {Literal::CreateR4<float>({{{{-3.f}, {-3.f}}, {{-1.f}, {-1.f}}},
                                 {{{1.f}, {1.f}}, {{3.f}, {3.f}}}})
           .get(),
       Literal::CreateR1<float>({0, 0}).get(),
       Literal::CreateR1<float>({16, 20}).get()});

  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.1));
}

}  // namespace
}  // namespace xla
