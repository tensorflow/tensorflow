/* Copyright 2017 The OpenXLA Authors.

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

// Tests that passing a bad shape to RNG's output parameter causes a validation
// failure rather than causing a crash.

#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/test.h"
#include "xla/shape.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(BadRngShapeValidationTest, DefaultConstructedShapeCreatesError) {
  XlaBuilder builder("BadRngShapeValidationTest");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto one = ConstantR0<float>(&builder, 1.0);
  RngUniform(zero, one, Shape());
  EXPECT_FALSE(builder.Build().ok());
}

TEST(BadRngShapeValidationTest, ShapeWithoutLayoutIsOk) {
  XlaBuilder builder("BadRngShapeValidationTest");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto one = ConstantR0<float>(&builder, 1.0);
  Shape shape;
  shape.set_element_type(F32);
  shape.add_dimensions(1);
  RngUniform(zero, one, shape);
  EXPECT_TRUE(builder.Build().ok());
}

}  // namespace
}  // namespace xla
