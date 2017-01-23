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

#include <vector>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class AxpySimpleTest : public ClientLibraryTestBase {};

TEST_F(AxpySimpleTest, AxTenValues) {
  ComputationBuilder builder(client_, "ax_10");
  auto alpha = builder.ConstantR0<float>(3.1415926535);
  auto x = builder.ConstantR1<float>(
      {-1.0, 1.0, 2.0, -2.0, -3.0, 3.0, 4.0, -4.0, -5.0, 5.0});
  auto ax = builder.Mul(alpha, x);

  std::vector<float> expected = {
      -3.14159265, 3.14159265,  6.28318531,   -6.28318531,  -9.42477796,
      9.42477796,  12.56637061, -12.56637061, -15.70796327, 15.70796327};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(AxpySimpleTest, AxpyZeroValues) {
  ComputationBuilder builder(client_, "axpy_10");
  auto alpha = builder.ConstantR0<float>(3.1415926535);
  auto x = builder.ConstantR1<float>({});
  auto y = builder.ConstantR1<float>({});
  auto ax = builder.Mul(alpha, x);
  auto axpy = builder.Add(ax, y);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(AxpySimpleTest, AxpyTenValues) {
  ComputationBuilder builder(client_, "axpy_10");
  auto alpha = builder.ConstantR0<float>(3.1415926535);
  auto x = builder.ConstantR1<float>(
      {-1.0, 1.0, 2.0, -2.0, -3.0, 3.0, 4.0, -4.0, -5.0, 5.0});
  auto y = builder.ConstantR1<float>(
      {5.0, -5.0, -4.0, 4.0, 3.0, -3.0, -2.0, 2.0, 1.0, -1.0});
  auto ax = builder.Mul(alpha, x);
  auto axpy = builder.Add(ax, y);

  std::vector<float> expected = {
      1.85840735, -1.85840735, 2.28318531,   -2.28318531,  -6.42477796,
      6.42477796, 10.56637061, -10.56637061, -14.70796327, 14.70796327};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
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
