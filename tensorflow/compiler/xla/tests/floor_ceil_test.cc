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

#include <limits>
#include <string>

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class FloorCeilTest : public ClientLibraryTestBase {
 public:
  enum Function {
    kFloor,
    kCeil,
  };

  // Runs a computation and comparison on expected vs f(input)
  void TestR1F32(tensorflow::gtl::ArraySlice<float> input,
                 tensorflow::gtl::ArraySlice<float> expected, Function f) {
    LOG(INFO) << "input: {" << tensorflow::str_util::Join(expected, ", ")
              << "}";
    ComputationBuilder builder(client_, TestName());
    auto c = builder.ConstantR1<float>(input);
    if (f == kCeil) {
      builder.Ceil(c);
    } else {
      ASSERT_EQ(kFloor, f);
      builder.Floor(c);
    }
    ComputeAndCompareR1<float>(&builder, expected, /*arguments=*/{});
  }

  void TestR0F32(float input, float expected, Function f) {
    LOG(INFO) << "input: " << expected;
    ComputationBuilder builder(client_, TestName());
    auto c = builder.ConstantR0<float>(input);
    if (f == kCeil) {
      builder.Ceil(c);
    } else {
      ASSERT_EQ(kFloor, f);
      builder.Floor(c);
    }
    ComputeAndCompareR0<float>(&builder, expected, /*arguments=*/{});
  }

  const ErrorSpec error_spec_{0.0001};

  float infinity_ = std::numeric_limits<float>::infinity();
  float minus_infinity_ = -std::numeric_limits<float>::infinity();
};

// Interesting notes:
// * if you pass snan the CPU doesn't canonicalize it to qnan.
// * passing x86-based CPU's qnan to the GPU makes a different nan
//   "7fc00000=nan=nan vs 7fffffff=nan=nan"

XLA_TEST_F(FloorCeilTest, R1S0Floor) { TestR1F32({}, {}, kFloor); }

TEST_F(FloorCeilTest, R1Floor) {
  TestR1F32({0.0, -0.0, infinity_, minus_infinity_, 1.1, -0.1},
            {0.0, -0.0, infinity_, minus_infinity_, 1.0, -1.0}, kFloor);
}

TEST_F(FloorCeilTest, R1Ceil) {
  TestR1F32({0.0, -0.0, infinity_, minus_infinity_, 1.1, -0.1},
            {0.0, -0.0, infinity_, minus_infinity_, 2.0, -0.0}, kCeil);
}

TEST_F(FloorCeilTest, R0Floor) {
  TestR0F32(0.0, 0.0, kFloor);
  TestR0F32(-0.0, -0.0, kFloor);
  TestR0F32(infinity_, infinity_, kFloor);
  TestR0F32(minus_infinity_, minus_infinity_, kFloor);
  TestR0F32(1.1, 1.0, kFloor);
  TestR0F32(-0.1, -1.0, kFloor);
}

TEST_F(FloorCeilTest, R0Ceil) {
  TestR0F32(0.0, 0.0, kCeil);
  TestR0F32(-0.0, -0.0, kCeil);
  TestR0F32(infinity_, infinity_, kCeil);
  TestR0F32(minus_infinity_, minus_infinity_, kCeil);
  TestR0F32(1.1, 2.0, kCeil);
  TestR0F32(-0.1, -0.0, kCeil);
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
