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

#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
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
  ComputationBuilder builder(client_, TestName());
  auto zero = builder.ConstantR0<float>(0.0);
  auto one = builder.ConstantR0<float>(1.0);
  Shape default_constructed;
  builder.RngUniform(zero, one, default_constructed);

  StatusOr<Computation> computation = builder.Build();
  EXPECT_FALSE(computation.ok());
  LOG(INFO) << "status received: " << computation.status();
  EXPECT_THAT(computation.status().error_message(),
              ::testing::HasSubstr("shape has invalid"));
}

TEST_F(BadRngShapeValidationTest, ShapeWithoutLayoutIsOk) {
  ComputationBuilder builder(client_, TestName());
  auto zero = builder.ConstantR0<float>(0.0);
  auto one = builder.ConstantR0<float>(1.0);
  Shape sans_layout;
  sans_layout.set_element_type(F32);
  sans_layout.add_dimensions(1);

  builder.RngUniform(zero, one, sans_layout);

  StatusOr<Computation> computation = builder.Build();
  ASSERT_TRUE(computation.ok());
  LOG(INFO) << computation.status();
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
