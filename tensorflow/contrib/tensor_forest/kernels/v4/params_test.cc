// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/platform/test.h"

namespace {

using tensorflow::tensorforest::DepthDependentParam;
using tensorflow::tensorforest::ResolveParam;

TEST(ParamsTest, TestConstant) {
  DepthDependentParam param;
  param.set_constant_value(10.0);

  ASSERT_EQ(ResolveParam(param, 0), 10.0);
  ASSERT_EQ(ResolveParam(param, 100), 10.0);
}

TEST(ParamsTest, TestLinear) {
  DepthDependentParam param;
  auto* linear = param.mutable_linear();
  linear->set_y_intercept(100.0);
  linear->set_slope(-10.0);
  linear->set_min_val(23.0);
  linear->set_max_val(90.0);

  ASSERT_EQ(ResolveParam(param, 0), 90);
  ASSERT_EQ(ResolveParam(param, 1), 90);
  ASSERT_EQ(ResolveParam(param, 2), 80);

  ASSERT_EQ(ResolveParam(param, 30), 23);
}

TEST(ParamsTest, TestExponential) {
  DepthDependentParam param;
  auto* expo = param.mutable_exponential();
  expo->set_bias(100.0);
  expo->set_base(10.0);
  expo->set_multiplier(-1.0);
  expo->set_depth_multiplier(1.0);

  ASSERT_EQ(ResolveParam(param, 0), 99);
  ASSERT_EQ(ResolveParam(param, 1), 90);
  ASSERT_EQ(ResolveParam(param, 2), 0);
}

TEST(ParamsTest, TestThreshold) {
  DepthDependentParam param;
  auto* threshold = param.mutable_threshold();
  threshold->set_on_value(100.0);
  threshold->set_off_value(10.0);
  threshold->set_threshold(5.0);

  ASSERT_EQ(ResolveParam(param, 0), 10);
  ASSERT_EQ(ResolveParam(param, 4), 10);
  ASSERT_EQ(ResolveParam(param, 5), 100);
  ASSERT_EQ(ResolveParam(param, 6), 100);
}

}  // namespace
