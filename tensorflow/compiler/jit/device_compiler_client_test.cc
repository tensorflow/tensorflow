/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_compiler_client.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace {

TEST(GetExecutableOptionTest, Basic) {
  XlaCompiler::Options options;
  options.device_ordinal = 0;
  options.alias_passthrough_params = true;
  options.detailed_logging = true;
  XlaCompiler::CompilationResult result;
  xla::Shape xla_output_shape;
  result.xla_output_shape = xla_output_shape;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_EQ(build_option.device_ordinal(), 0);
  EXPECT_EQ(build_option.result_layout()->ToString(),
            xla_output_shape.ToString());
  EXPECT_EQ(build_option.alias_passthrough_params(), true);
  EXPECT_EQ(build_option.debug_options().xla_detailed_logging(), true);
  EXPECT_EQ(build_option.debug_options().xla_enable_dumping(), true);
}

TEST(GetExecutableOptionTest, DefaultDeviceOrdinal) {
  XlaCompiler::Options options;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/0);

  EXPECT_EQ(build_option.device_ordinal(), 0);
}

TEST(GetExecutableOptionTest, DeviceOrdinalNotSet) {
  XlaCompiler::Options options;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_EQ(build_option.device_ordinal(), -1);
}

TEST(GetExecutableOptionTest, DumpingWithoutDetailedLogging) {
  XlaCompiler::Options options;
  options.detailed_logging = false;
  XlaCompiler::CompilationResult result;

  auto build_option =
      GetExecutableBuildOptions(options, result, /*default_device_ordinal=*/-1);

  EXPECT_FALSE(build_option.debug_options().xla_detailed_logging());
  EXPECT_TRUE(build_option.debug_options().xla_enable_dumping());
}

}  // namespace
}  // namespace tensorflow
