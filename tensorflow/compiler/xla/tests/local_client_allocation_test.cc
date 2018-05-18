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

#include <memory>

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class LocalClientAllocationTest : public LocalClientTestBase {
 protected:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(LocalClientAllocationTest, AddVectors) {
  XlaBuilder builder(TestName());
  auto x = builder.ConstantR1<float>({0.0f, 1.0f, 2.0f});
  auto y = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  builder.Add(x, y);

  TestAllocator* allocator = GetOrCreateAllocator(local_client_->platform());

  auto x_array =
      LiteralToShapedBuffer(*Literal::CreateR1<float>({0.0f, 1.0f, 2.0f}));

  int64 allocation_count_before = allocator_->allocation_count();

  // Override the allocator via 'options'. Tests that allocation and
  // deallocation happen on the right allocator.
  ExecutableRunOptions options;
  options.set_allocator(allocator);
  tensorflow::gtl::optional<ScopedShapedBuffer> result =
      ExecuteLocallyOrDie(builder.Build().ValueOrDie(), {},
                          DefaultExecutableBuildOptions(), options);

  LiteralTestUtil::ExpectR1Near<float>(
      {2.0f, 4.0f, 6.0f}, *ShapedBufferToLiteral(*result), error_spec_);

  // At least one allocation should have been performed when executing the
  // computation.
  EXPECT_GT(allocator_->allocation_count(), allocation_count_before);

  // Deallocate result and verify that deallocate was called once.
  int64 deallocation_count_before = allocator_->deallocation_count();
  result.reset();
  EXPECT_EQ(deallocation_count_before + 1, allocator_->deallocation_count());
}

XLA_TEST_F(LocalClientAllocationTest, RunOnDevices) {
  // Run a computation on every device on the system. Verify that allocation
  // occurs on the proper device.
  XlaBuilder builder(TestName());
  auto x = builder.ConstantR1<float>({0.0f, 1.0f, 2.0f});
  auto y = builder.ConstantR1<float>({2.0f, 3.0f, 4.0f});
  builder.Add(x, y);
  auto computation = builder.Build().ConsumeValueOrDie();

  TestAllocator* allocator = GetOrCreateAllocator(local_client_->platform());
  for (int d = 0; d < local_client_->device_count(); ++d) {
    if (!local_client_->device_ordinal_supported(d)) {
      continue;
    }

    int64 device_allocation_count_before = allocator->allocation_count(d);
    int64 allocation_count_before = allocator->allocation_count();

    auto result = ExecuteLocallyOrDie(
        computation, {}, ExecutableBuildOptions().set_device_ordinal(d),
        ExecutableRunOptions().set_device_ordinal(d).set_allocator(allocator));
    LiteralTestUtil::ExpectR1Near<float>(
        {2.0f, 4.0f, 6.0f}, *ShapedBufferToLiteral(result), error_spec_);

    // At least one allocation should have been performed when executing the
    // computation.
    EXPECT_GT(allocator->allocation_count(), allocation_count_before);
    EXPECT_GT(allocator->allocation_count(d), device_allocation_count_before);
  }
}

}  // namespace
}  // namespace xla
