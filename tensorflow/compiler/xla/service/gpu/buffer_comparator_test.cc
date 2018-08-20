/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"

#include <limits>
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class BufferComparatorTest : public testing::Test {
 protected:
  BufferComparatorTest()
      : backend_(Backend::CreateDefaultBackend().ConsumeValueOrDie()),
        stream_exec_(backend_->default_stream_executor()),
        allocator_(stream_exec_->platform(), {stream_exec_}),
        compiler_(Compiler::GetForPlatform(stream_exec_->platform())
                      .ConsumeValueOrDie()) {}

  // Take floats only for convenience. Still uses half internally.
  bool CompareEqualFloatBuffers(const std::vector<float>& lhs_float,
                                const std::vector<float>& rhs_float) {
    std::vector<half> lhs(lhs_float.begin(), lhs_float.end());
    std::vector<half> rhs(rhs_float.begin(), rhs_float.end());
    se::Stream stream(stream_exec_);
    stream.Init();

    auto owning_lhs_buffer =
        allocator_
            .Allocate(stream_exec_->device_ordinal(), lhs.size() * sizeof(half))
            .ConsumeValueOrDie();

    auto owning_rhs_buffer =
        allocator_
            .Allocate(stream_exec_->device_ordinal(), rhs.size() * sizeof(half))
            .ConsumeValueOrDie();

    auto lhs_buffer =
        se::DeviceMemory<Eigen::half>(owning_lhs_buffer.AsDeviceMemoryBase());
    auto rhs_buffer =
        se::DeviceMemory<Eigen::half>(owning_rhs_buffer.AsDeviceMemoryBase());

    stream.ThenMemcpy(&lhs_buffer, lhs.data(), lhs_buffer.size());
    stream.ThenMemcpy(&rhs_buffer, rhs.data(), rhs_buffer.size());

    TF_CHECK_OK(stream.BlockHostUntilDone());

    return F16BufferComparator::Create(lhs_buffer, compiler_, &allocator_,
                                       &stream)
        .ConsumeValueOrDie()
        .CompareEqual(rhs_buffer)
        .ConsumeValueOrDie();
  }

  std::unique_ptr<Backend> backend_;
  se::StreamExecutor* stream_exec_;
  StreamExecutorMemoryAllocator allocator_;
  Compiler* compiler_;
};

TEST_F(BufferComparatorTest, TestNaNs) {
  EXPECT_TRUE(CompareEqualFloatBuffers({std::nanf("")}, {std::nanf("")}));
  // NaN values with different bit patterns should compare equal.
  EXPECT_TRUE(CompareEqualFloatBuffers({std::nanf("")}, {std::nanf("1234")}));
  EXPECT_FALSE(CompareEqualFloatBuffers({std::nanf("")}, {1.}));
}

TEST_F(BufferComparatorTest, TestInfs) {
  const auto inf = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(CompareEqualFloatBuffers({inf}, {std::nanf("")}));
  EXPECT_TRUE(CompareEqualFloatBuffers({inf}, {inf}));
  EXPECT_TRUE(CompareEqualFloatBuffers({inf}, {65504}));
  EXPECT_TRUE(CompareEqualFloatBuffers({-inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers({inf}, {-65504}));
  EXPECT_FALSE(CompareEqualFloatBuffers({-inf}, {65504}));

  EXPECT_FALSE(CompareEqualFloatBuffers({inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers({inf}, {-20}));
  EXPECT_FALSE(CompareEqualFloatBuffers({-inf}, {20}));
  EXPECT_FALSE(CompareEqualFloatBuffers({-inf}, {-20}));
}

TEST_F(BufferComparatorTest, TestNumbers) {
  EXPECT_TRUE(CompareEqualFloatBuffers({20}, {20.1}));
  EXPECT_FALSE(CompareEqualFloatBuffers({0}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers({0.9}, {1}));
  EXPECT_TRUE(CompareEqualFloatBuffers({9}, {10}));
  EXPECT_TRUE(CompareEqualFloatBuffers({10}, {9}));
}

TEST_F(BufferComparatorTest, TestMultiple) {
  EXPECT_TRUE(CompareEqualFloatBuffers({20, 30, 40, 50, 60},
                                       {20.1, 30.1, 40.1, 50.1, 60.1}));
  std::vector<float> lhs(200);
  std::vector<float> rhs(200);
  for (int i = 0; i < 200; i++) {
    EXPECT_TRUE(CompareEqualFloatBuffers(lhs, rhs))
        << "should be the same at index " << i;
    lhs[i] = 3;
    rhs[i] = 5;
    EXPECT_FALSE(CompareEqualFloatBuffers(lhs, rhs))
        << "should be the different at index " << i;
    lhs[i] = 0;
    rhs[i] = 0;
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
