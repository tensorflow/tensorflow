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

#ifndef XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
#define XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_computation.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/local_service.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/manifest_checking_test.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/xla_data.pb.h"

namespace xla {

class TestAllocator : public se::StreamExecutorMemoryAllocator {
 public:
  explicit TestAllocator(se::Platform* platform)
      : se::StreamExecutorMemoryAllocator(
            platform, PlatformUtil::GetStreamExecutors(platform).value()) {}

  absl::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;
  Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override;

  // Return the number of allocations that have been performed.
  int64_t allocation_count() const;
  int64_t allocation_count(int device_ordinal) const;

  // Return the number of deallocations that have been performed.
  int64_t deallocation_count() const;
  int64_t deallocation_count(int device_ordinal) const;

 private:
  mutable absl::Mutex count_mutex_;

  // Global counts of allocations and deallocations.
  int64_t allocation_count_ ABSL_GUARDED_BY(count_mutex_) = 0;
  int64_t deallocation_count_ ABSL_GUARDED_BY(count_mutex_) = 0;

  // Per-device counts of allocations and deallocations.
  std::map<int, int64_t> device_allocation_count_ ABSL_GUARDED_BY(count_mutex_);
  std::map<int, int64_t> device_deallocation_count_
      ABSL_GUARDED_BY(count_mutex_);
};

// A base class for tests which exercise the LocalClient interface.
class LocalClientTestBase : public ManifestCheckingTest {
 protected:
  struct EigenThreadPoolWrapper;
  explicit LocalClientTestBase(se::Platform* platform = nullptr);
  virtual ~LocalClientTestBase();

  static TestAllocator* GetOrCreateAllocator(se::Platform* platform);

  // Copy the given literal onto the default device and return a
  // ScopedShapedBuffer. Convenience wrapper around
  // LocalClient::LiteralToShapedBuffer.
  ScopedShapedBuffer LiteralToShapedBuffer(const Literal& literal);

  // Construct and return a literal containing the array represented by
  // shaped_buffer.
  Literal ShapedBufferToLiteral(const ShapedBuffer& shaped_buffer);

  // Execute the given computation on the local client. With and without
  // options.
  absl::StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments);
  absl::StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments);
  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      absl::Span<const ShapedBuffer* const> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  // Parses the given string and returns module as a VerifiedHloModule.
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text);
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
  ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                               const HloModuleConfig& config);

  // Returns a default set of execute options.
  ExecutableBuildOptions DefaultExecutableBuildOptions() const;

  // Returns a default set of execute options, configured to use allocator_
  // as the allocator.
  ExecutableRunOptions DefaultExecutableRunOptions() const;

  std::string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  // The allocator must live as long as the service, which lives until the end
  // of the process. So make the allocator static.
  static TestAllocator* allocator_;

  se::StreamExecutor* stream_executor_;
  TransferManager* transfer_manager_;

  LocalClient* local_client_;

  std::unique_ptr<EigenThreadPoolWrapper> thread_pool_wrapper_;
};

}  // namespace xla

#endif  // XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
