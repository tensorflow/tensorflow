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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class TestAllocator : public StreamExecutorMemoryAllocator {
 public:
  explicit TestAllocator(se::Platform* platform)
      : StreamExecutorMemoryAllocator(
            platform, PlatformUtil::GetStreamExecutors(platform).ValueOrDie()) {
  }

  StatusOr<se::DeviceMemoryBase> Allocate(int device_ordinal, uint64 size,
                                          bool retry_on_failure) override;
  tensorflow::Status Deallocate(int device_ordinal,
                                se::DeviceMemoryBase* mem) override;

  // Return the number of allocations that have been performed.
  int64 allocation_count() const;
  int64 allocation_count(int device_ordinal) const;

  // Return the number of deallocations that have been performed.
  int64 deallocation_count() const;
  int64 deallocation_count(int device_ordinal) const;

 private:
  mutable tensorflow::mutex count_mutex_;

  // Global counts of allocations and deallocations.
  int64 allocation_count_ GUARDED_BY(count_mutex_) = 0;
  int64 deallocation_count_ GUARDED_BY(count_mutex_) = 0;

  // Per-device counts of allocations and deallocations.
  std::map<int, int64> device_allocation_count_ GUARDED_BY(count_mutex_);
  std::map<int, int64> device_deallocation_count_ GUARDED_BY(count_mutex_);
};

// A base class for tests which exercise the LocalClient interface.
class LocalClientTestBase : public ::testing::Test {
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
  std::unique_ptr<Literal> ShapedBufferToLiteral(
      const ShapedBuffer& shaped_buffer);

  // Execute the given computation on the local client. With and without
  // options.
  StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments);
  StatusOr<ScopedShapedBuffer> ExecuteLocally(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments);
  ScopedShapedBuffer ExecuteLocallyOrDie(
      const XlaComputation& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const ExecutableBuildOptions& build_options,
      const ExecutableRunOptions& run_options);

  // Returns a default set of execute options.
  ExecutableBuildOptions DefaultExecutableBuildOptions() const;

  // Returns a default set of execute options, configured to use allocator_
  // as the allocator.
  ExecutableRunOptions DefaultExecutableRunOptions() const;

  string TestName() const {
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

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_LOCAL_CLIENT_TEST_BASE_H_
