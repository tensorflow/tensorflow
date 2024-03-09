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

#include <memory>

#include "xla/client/local_client.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/statusor.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

class BadRngShapeValidationTest : public ClientLibraryTestBase {};

TEST_F(BadRngShapeValidationTest, DefaultConstructedShapeCreatesError) {
  XlaBuilder builder(TestName());
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto one = ConstantR0<float>(&builder, 1.0);
  Shape default_constructed;
  RngUniform(zero, one, default_constructed);

  absl::StatusOr<XlaComputation> computation = builder.Build();
  EXPECT_FALSE(computation.ok());
  LOG(INFO) << "status received: " << computation.status();
  EXPECT_THAT(computation.status().message(),
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

  absl::StatusOr<XlaComputation> computation = builder.Build();
  ASSERT_TRUE(computation.ok());
  LOG(INFO) << computation.status();
}

}  // namespace
}  // namespace xla
