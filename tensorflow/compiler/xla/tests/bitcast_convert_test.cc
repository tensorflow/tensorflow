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

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
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

class BitcastConvertTest : public ClientLibraryTestBase {
 public:
  explicit BitcastConvertTest(se::Platform* platform = nullptr)
      : ClientLibraryTestBase(platform) {
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    mutable_debug_options()->add_xla_disable_hlo_passes("inline");
  }
};

TEST_F(BitcastConvertTest, ConvertR1S32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32>(&builder, {42, 64});
  BitcastConvertType(a, S32);

  std::vector<int32> expected = {42, 64};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, BitcastR1S32ToR1F32) {
  XlaBuilder builder(TestName());
  auto a =
      ConstantR1<int32>(&builder, {0, static_cast<int32>(0x80000000),
                                   0x3F800000, static_cast<int32>(0xBF800000),
                                   0x3F000000, static_cast<int32>(0xBF000000)});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

XLA_TEST_F(BitcastConvertTest, ConvertR1S0S32ToR1S0F32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32>(&builder, {});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertR1F32ToR1S32) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<float>(&builder, {42.6, 64.4});
  BitcastConvertType(a, S32);

  std::vector<int32> expected = {0x422a6666, 0x4280cccd};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertS32Extremes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<int32>(&builder, {std::numeric_limits<int32>::min(),
                                        std::numeric_limits<int32>::max()});
  BitcastConvertType(a, F32);

  std::vector<float> expected = {-0.0f, NAN};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0, 0));
}

TEST_F(BitcastConvertTest, ConvertMapToS32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "in");
  BitcastConvertType(param, S32);
  auto a = ConstantR1<float>(&builder, {42.0f, 64.0f});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<int32> expected = {0x42280000, 0x42800000};
  ComputeAndCompareR1<int32>(&builder, expected, {});
}

TEST_F(BitcastConvertTest, ConvertMapToF32) {
  XlaBuilder builder(TestName());
  auto b = builder.CreateSubBuilder("convert");
  auto param = Parameter(b.get(), 0, ShapeUtil::MakeShape(S32, {}), "in");
  BitcastConvertType(param, F32);
  auto a = ConstantR1<int32>(&builder, {0x42280000, 0x42800000});
  Map(&builder, {a}, b->BuildAndNoteError(), {0});

  std::vector<float> expected = {42.0f, 64.0f};
  ComputeAndCompareR1<float>(&builder, expected, {});
}

// Regression test for b/31758660. When ReshapeMover transforms
//   input -> reshape -> convert
// to
//   input -> convert -> reshape
// the new convert should have the same element type as the old convert.
TEST_F(BitcastConvertTest, ConvertReshape) {
  XlaBuilder builder(TestName());
  auto input = ConstantR1<int32>(&builder, {0x42280000});
  auto reshape = Reshape(input, /*dimensions=*/{0}, /*new_sizes=*/{});
  BitcastConvertType(reshape, F32);

  ComputeAndCompareR0<float>(&builder, 42.0f, {});
}

}  // namespace
}  // namespace xla
