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

// Tests that passing a bad shape to RNG's output parameter causes a validation
// failure rather than causing a crash.

#include <memory>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

class BadRngShapeValidationTest : public ClientLibraryTestBase {};

TEST_F(BadRngShapeValidationTest, DefaultConstructedShapeCreatesError) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto one = ConstantR0<float>(&builder, 1.0);
  Shape default_constructed;
  RngUniform(zero, one, default_constructed);

  StatusOr<XlaComputation> computation = builder.Build();
  EXPECT_FALSE(computation.ok());
  LOG(INFO) << "status received: " << computation.status();
  EXPECT_THAT(computation.status().error_message(),
              ::testing::HasSubstr("shape has invalid"));
}

TEST_F(BadRngShapeValidationTest, ShapeWithoutLayoutIsOk) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto one = ConstantR0<float>(&builder, 1.0);
  Shape sans_layout;
  sans_layout.set_element_type(F32);
  sans_layout.add_dimensions(1);

  RngUniform(zero, one, sans_layout);

  StatusOr<XlaComputation> computation = builder.Build();
  ASSERT_TRUE(computation.ok());
  LOG(INFO) << computation.status();
}

}  // namespace
}  // namespace xla
