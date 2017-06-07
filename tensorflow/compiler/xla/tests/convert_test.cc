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

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class ConvertTest : public ClientLibraryTestBase {
 public:
  explicit ConvertTest(perftools::gputools::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }
};

TEST_F(ConvertTest, ConvertR1S32ToR1S32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({42, 64});
  builder.ConvertElementType(a, S32);

  std::vector<int32> expected = {42, 64};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertR1F32ToR1F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({42.0f, 64.0f});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ConvertTest, ConvertR1S32ToR1F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({42, 64});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ConvertTest, ConvertR1S0S32ToR1S0F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>({});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ConvertTest, ConvertR1F32ToR1S32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({42.6, 64.4});
  builder.ConvertElementType(a, S32);

  std::vector<int32> expected = {42, 64};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1S64ToR1F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int64>({32, 64});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {32.0, 64.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint8_t>({32, 64});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {32.0, 64.0};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1S32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint8_t>({32, 64});
  builder.ConvertElementType(a, S32);

  std::vector<int32_t> expected = {32, 64};
  ComputeAndCompareR1<int32_t>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1U8ToR1U32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<uint8_t>({32, 64});
  builder.ConvertElementType(a, U32);

  std::vector<uint32_t> expected = {32, 64};
  ComputeAndCompareR1<uint32_t>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1F32ToR1F64) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<float>({32.0f, 64.0f});
  builder.ConvertElementType(a, F64);

  std::vector<double> expected = {32.0, 64.0};
  ComputeAndCompareR1<double>(&builder, expected, {});
}

XLA_TEST_F(ConvertTest, ConvertR1F64ToR1F32) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<double>({32.0, 64.0});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {32.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertS32Extremes) {
  ComputationBuilder builder(client_, TestName());
  auto a = builder.ConstantR1<int32>(
      {std::numeric_limits<int32>::min(), std::numeric_limits<int32>::max()});
  builder.ConvertElementType(a, F32);

  std::vector<float> expected = {
      static_cast<float>(std::numeric_limits<int32>::min()),
      static_cast<float>(std::numeric_limits<int32>::max())};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ConvertTest, ConvertMapToS32) {
  ComputationBuilder builder(client_, TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = b->Parameter(0, ShapeUtil::MakeShape(F32, {}), "in");
  b->ConvertElementType(param, S32);
  auto a = builder.ConstantR1<float>({42.0f, 64.0f});
  builder.Map({a}, b->BuildAndNoteError());

  std::vector<int32> expected = {42, 64};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

TEST_F(ConvertTest, ConvertMapToF32) {
  ComputationBuilder builder(client_, TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = b->Parameter(0, ShapeUtil::MakeShape(S32, {}), "in");
  b->ConvertElementType(param, F32);
  auto a = builder.ConstantR1<int32>({42, 64});
  builder.Map({a}, b->BuildAndNoteError());

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Regression test for b/31758660. When ReshapeMover transforms
//   input -> reshape -> convert
// to
//   input -> convert -> reshape
// the new convert should have the same element type as the old convert.
TEST_F(ConvertTest, ConvertReshape) {
  ComputationBuilder builder(client_, TestName());
  auto input = builder.ConstantR1<int32>({42});
  auto reshape = builder.Reshape(input, /*dimensions=*/{0}, /*new_sizes=*/{});
  builder.ConvertElementType(reshape, F32);

  ComputeAndCompareR0<float>(&builder, 42.0f, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
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
