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

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/service/local_service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/local_client_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class LocalClientAllocationTest : public LocalClientTestBase {
 protected:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(LocalClientAllocationTest, AddVectors) {
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {0.0f, 1.0f, 2.0f});
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);

  TestAllocator* allocator = GetOrCreateAllocator(local_client_->platform());

  auto x_array =
      LiteralToShapedBuffer(LiteralUtil::CreateR1<float>({0.0f, 1.0f, 2.0f}));

  int64_t allocation_count_before = allocator_->allocation_count();

  // Override the allocator via 'options'. Tests that allocation and
  // deallocation happen on the right allocator.
  ExecutableRunOptions options;
  options.set_allocator(allocator);
  std::optional<ScopedShapedBuffer> result = ExecuteLocallyOrDie(
      builder.Build().value(), {}, DefaultExecutableBuildOptions(), options);

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(*result), error_spec_);

  // At least one allocation should have been performed when executing the
  // computation.
  EXPECT_GT(allocator_->allocation_count(), allocation_count_before);

  // Deallocate result and verify that deallocate was called once.
  int64_t deallocation_count_before = allocator_->deallocation_count();
  result.reset();
  EXPECT_EQ(deallocation_count_before + 1, allocator_->deallocation_count());
}

XLA_TEST_F(LocalClientAllocationTest, RunOnDevices) {
  // Run a computation on every device on the system. Verify that allocation
  // occurs on the proper device.
  XlaBuilder builder(TestName());
  auto x = ConstantR1<float>(&builder, {0.0f, 1.0f, 2.0f});
  auto y = ConstantR1<float>(&builder, {2.0f, 3.0f, 4.0f});
  Add(x, y);
  auto computation = builder.Build().value();

  TestAllocator* allocator = GetOrCreateAllocator(local_client_->platform());
  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      continue;
    }

    int64_t device_allocation_count_before = allocator->allocation_count(d);
    int64_t allocation_count_before = allocator->allocation_count();

    auto result = ExecuteLocallyOrDie(
        computation, {}, ExecutableBuildOptions().set_device_ordinal(d),
        ExecutableRunOptions().set_device_ordinal(d).set_allocator(allocator));
    LiteralTestUtil::ExpectR1Near<float>(
        {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(result), error_spec_);

    // At least one allocation should have been performed when executing the
    // computation.
    EXPECT_GT(allocator->allocation_count(), allocation_count_before);
    EXPECT_GT(allocator->allocation_count(d), device_allocation_count_before);
  }
}

}  // namespace
}  // namespace xla
