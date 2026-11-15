/* Copyright 2026 The OpenXLA Authors.

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

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/service/backend.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/stream_pool.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/cuda/green_context.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class CallbackThunk : public Thunk {
 public:
  using Callback = std::function<absl::Status(const ExecuteParams&)>;

  explicit CallbackThunk(Callback callback)
      : Thunk(kKernel, ThunkInfo()), callback_(std::move(callback)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return callback_(params);
  }

  BufferUses buffer_uses() const override { return {}; }

  absl::StatusOr<ThunkProto> ToProto() const override {
    return absl::UnimplementedError("CallbackThunk is test-only");
  }

 private:
  Callback callback_;
};

Thunk::ThunkInfo ThunkInfoWithId(int id) {
  Thunk::ThunkInfo info;
  info.thunk_id = id;
  return info;
}

void AppendLocalityRegion(ThunkSequence& sequence, int thunk_id, int domain,
                          ThunkSequence body) {
  auto start = std::make_unique<AsyncStartThunk>(
      ThunkInfoWithId(thunk_id), LocalityDomainStreamId(domain),
      std::move(body), /*devices_per_host=*/1);
  std::shared_ptr<AsyncExecution> execution = start->async_execution();
  sequence.push_back(std::move(start));
  sequence.Emplace<AsyncDoneThunk>(ThunkInfoWithId(thunk_id + 1000),
                                   std::move(execution));
}

class TrackingLocalityStreamBorrower {
 public:
  TrackingLocalityStreamBorrower(se::StreamExecutor* executor,
                                 int locality_domain_count) {
    pools_.reserve(locality_domain_count);
    for (int i = 0; i < locality_domain_count; ++i) {
      pools_.push_back(std::make_unique<StreamPool>(executor));
    }
  }

  absl::StatusOr<StreamPool::Ptr> Borrow(int device_ordinal, int domain,
                                         se::StreamPriority priority) {
    ++borrow_count_;
    borrowed_device_ordinals_.push_back(device_ordinal);
    borrowed_domains_.push_back(domain);
    if (!borrow_error_.ok()) {
      return borrow_error_;
    }
    if (domain < 0 || domain >= static_cast<int>(pools_.size())) {
      return absl::InvalidArgumentError(
          "tracking borrower domain out of range");
    }
    ABSL_ASSIGN_OR_RETURN(StreamPool::Ptr stream,
                     pools_[domain]->BorrowStream(priority));
    last_borrowed_stream_ = stream.get();
    return stream;
  }

  void set_borrow_error(absl::Status status) {
    borrow_error_ = std::move(status);
  }

  int borrow_count() const { return borrow_count_; }
  const std::vector<int>& borrowed_device_ordinals() const {
    return borrowed_device_ordinals_;
  }
  const std::vector<int>& borrowed_domains() const { return borrowed_domains_; }
  se::Stream* last_borrowed_stream() const { return last_borrowed_stream_; }
  StreamPool* pool(int domain) { return pools_[domain].get(); }

 private:
  std::vector<std::unique_ptr<StreamPool>> pools_;
  absl::Status borrow_error_ = absl::OkStatus();
  int borrow_count_ = 0;
  std::vector<int> borrowed_device_ordinals_;
  std::vector<int> borrowed_domains_;
  se::Stream* last_borrowed_stream_ = nullptr;
};

class GpuExecutableLocalityDomainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(std::string platform_name,
                         PlatformUtil::CanonicalPlatformName("gpu"));
    ASSERT_OK_AND_ASSIGN(platform_,
                         se::PlatformManager::PlatformWithName(platform_name));
    ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    ASSERT_OK_AND_ASSIGN(main_stream_, executor_->CreateStream());
    allocator_ =
        std::make_unique<se::StreamExecutorAddressAllocator>(executor_);
  }

  absl::StatusOr<std::unique_ptr<GpuExecutable>> CreateExecutable(
      ThunkSequence thunks, int compiled_locality_domain_count) {
    se::DeviceDescription device_description =
        executor_->GetDeviceDescription();
    device_description.set_locality_domain_count(
        compiled_locality_domain_count);

    GpuExecutable::Params params;
    params.module_name = "gpu_executable_locality_domain_test";
    params.executable = std::make_unique<ThunkExecutor>(std::move(thunks));
    params.device_description = std::move(device_description);
    params.enable_debug_info_manager = false;
    params.buffer_assignment_proto = BufferAssignmentProto();
    params.buffer_allocations_debug_summary = "empty test assignment";
    return GpuExecutable::Create(std::move(params));
  }

  ServiceExecutableRunOptions MakeRunOptions(
      ServiceExecutableRunOptions::LocalityDomainStreamBorrower borrower =
          nullptr,
      se::DeviceAddressAllocator* allocator = nullptr,
      se::Stream* stream = nullptr) {
    ExecutableRunOptions options;
    options.set_stream(stream == nullptr ? main_stream_.get() : stream)
        .set_allocator(allocator == nullptr ? allocator_.get() : allocator)
        .set_device_ordinal(executor_->device_ordinal())
        .set_local_device_count(1);
    return ServiceExecutableRunOptions(options, /*stream_borrower=*/nullptr,
                                       std::move(borrower));
  }

  BufferAllocations EmptyAllocations(
      se::DeviceAddressAllocator* allocator = nullptr) {
    return BufferAllocations(
        {}, executor_->device_ordinal(),
        allocator == nullptr ? allocator_.get() : allocator);
  }

  int runtime_locality_domain_count() const {
    return executor_->GetDeviceDescription().locality_domain_count();
  }

  se::Platform* platform_ = nullptr;
  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Stream> main_stream_;
  std::unique_ptr<se::StreamExecutorAddressAllocator> allocator_;
};

TEST_F(GpuExecutableLocalityDomainTest,
       NoReferencesDoNotInvokeLocalityBorrower) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(ThunkSequence(), /*count=*/0));
  int borrow_count = 0;
  ServiceExecutableRunOptions run_options = MakeRunOptions(
      [&](int, int, se::StreamPriority) -> absl::StatusOr<StreamPool::Ptr> {
        ++borrow_count;
        return absl::FailedPreconditionError("must not be called");
      });
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_OK(executable->ExecuteThunks(allocations, &run_options));
  EXPECT_EQ(borrow_count, 0);
}

TEST_F(GpuExecutableLocalityDomainTest,
       ReferencedDomainRequiresSupportedCounts) {
  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/0, ThunkSequence());

  // A zero compile/PjRT count is unsupported regardless of runtime hardware.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(std::move(thunks), /*count=*/0));
  ServiceExecutableRunOptions run_options = MakeRunOptions();
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(
      executable->ExecuteThunks(allocations, &run_options),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("compile/PjRT reports 0 usable locality domains")));
}

TEST_F(GpuExecutableLocalityDomainTest,
       NegativeCompileLocalityCountIsRejected) {
  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/0, ThunkSequence());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(std::move(thunks), /*count=*/-1));
  ServiceExecutableRunOptions run_options = MakeRunOptions();
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(
      executable->ExecuteThunks(allocations, &run_options),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("compile/PjRT reports -1 usable locality domains")));
}

TEST_F(GpuExecutableLocalityDomainTest,
       RuntimeWithoutLocalityDomainsReportsUnsupported) {
  if (runtime_locality_domain_count() != 0) {
    GTEST_SKIP() << "Runtime exposes locality domains";
  }

  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/0, ThunkSequence());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(std::move(thunks), /*count=*/1));
  ServiceExecutableRunOptions run_options = MakeRunOptions();
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(executable->ExecuteThunks(allocations, &run_options),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("runtime executor reports 0")));
}

TEST_F(GpuExecutableLocalityDomainTest, MissingBorrowerIsRejected) {
  if (runtime_locality_domain_count() == 0) {
    GTEST_SKIP() << "Requires a runtime locality domain";
  }

  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/0, ThunkSequence());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      CreateExecutable(std::move(thunks), runtime_locality_domain_count()));
  ServiceExecutableRunOptions run_options = MakeRunOptions();
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(
      executable->ExecuteThunks(allocations, &run_options),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("requires a locality-domain stream borrower")));
}

TEST_F(GpuExecutableLocalityDomainTest, OutOfRangeDomainIdIsRejected) {
  if (runtime_locality_domain_count() == 0) {
    GTEST_SKIP() << "Requires a runtime locality domain";
  }
  const int count = runtime_locality_domain_count();
  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/count,
                       ThunkSequence());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(std::move(thunks), count));
  ServiceExecutableRunOptions run_options = MakeRunOptions(
      [](int, int, se::StreamPriority) -> absl::StatusOr<StreamPool::Ptr> {
        return absl::InternalError("must be validated before borrowing");
      });
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(executable->ExecuteThunks(allocations, &run_options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid locality-domain stream id")));
}

TEST_F(GpuExecutableLocalityDomainTest,
       CompileRuntimeCountMismatchRejectsDomainOutsideSmallerCount) {
  const int runtime_count = runtime_locality_domain_count();
  if (runtime_count == 0) {
    GTEST_SKIP() << "Requires a runtime locality domain";
  }

  const int compiled_count = runtime_count == 1 ? 2 : 1;
  const int invalid_domain = 1;
  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, invalid_domain, ThunkSequence());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       CreateExecutable(std::move(thunks), compiled_count));
  ServiceExecutableRunOptions run_options = MakeRunOptions(
      [](int, int, se::StreamPriority) -> absl::StatusOr<StreamPool::Ptr> {
        return absl::InternalError("must be validated before borrowing");
      });
  BufferAllocations allocations = EmptyAllocations();

  absl::Status status = executable->ExecuteThunks(allocations, &run_options);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Invalid locality-domain stream id")));
  EXPECT_THAT(status.message(), HasSubstr("compile/PjRT reports"));
  EXPECT_THAT(status.message(), HasSubstr("runtime executor reports"));
}

TEST_F(GpuExecutableLocalityDomainTest, BorrowerErrorIsPropagatedWithContext) {
  if (runtime_locality_domain_count() == 0) {
    GTEST_SKIP() << "Requires a runtime locality domain";
  }

  ThunkSequence thunks;
  AppendLocalityRegion(thunks, /*thunk_id=*/1, /*domain=*/0, ThunkSequence());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      CreateExecutable(std::move(thunks), runtime_locality_domain_count()));
  TrackingLocalityStreamBorrower borrower(executor_,
                                          runtime_locality_domain_count());
  borrower.set_borrow_error(absl::UnavailableError("pool unavailable"));
  ServiceExecutableRunOptions run_options = MakeRunOptions(
      [&](int device_ordinal, int domain, se::StreamPriority priority) {
        return borrower.Borrow(device_ordinal, domain, priority);
      });
  BufferAllocations allocations = EmptyAllocations();

  EXPECT_THAT(
      executable->ExecuteThunks(allocations, &run_options),
      StatusIs(absl::StatusCode::kUnavailable,
               HasSubstr("Failed to borrow locality-domain stream 0: pool "
                         "unavailable")));
  EXPECT_EQ(borrower.borrow_count(), 1);
}

TEST_F(GpuExecutableLocalityDomainTest,
       NestedDuplicateDomainsDeduplicateAndKeepLeaseAlive) {
  if (runtime_locality_domain_count() == 0) {
    GTEST_SKIP() << "Requires a runtime locality domain";
  }

  TrackingLocalityStreamBorrower borrower(executor_,
                                          runtime_locality_domain_count());
  std::vector<se::Stream*> executed_streams;
  bool lease_was_live_during_execution = false;

  auto record_stream = [&](const Thunk::ExecuteParams& params) {
    executed_streams.push_back(params.stream);
    ABSL_ASSIGN_OR_RETURN(
        StreamPool::Ptr concurrent,
        borrower.pool(/*domain=*/0)->BorrowStream(se::StreamPriority::Default));
    lease_was_live_during_execution =
        lease_was_live_during_execution ||
        concurrent.get() != borrower.last_borrowed_stream();
    return absl::OkStatus();
  };

  ThunkSequence nested;
  AppendLocalityRegion(nested, /*thunk_id=*/1, /*domain=*/0,
                       ThunkSequence::Of<CallbackThunk>(record_stream));
  AppendLocalityRegion(nested, /*thunk_id=*/2, /*domain=*/0,
                       ThunkSequence::Of<CallbackThunk>(record_stream));
  ThunkSequence thunks = ThunkSequence::Of<SequentialThunk>(
      ThunkInfoWithId(100), std::move(nested));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      CreateExecutable(std::move(thunks), runtime_locality_domain_count()));
  ServiceExecutableRunOptions run_options = MakeRunOptions(
      [&](int device_ordinal, int domain, se::StreamPriority priority) {
        return borrower.Borrow(device_ordinal, domain, priority);
      });
  BufferAllocations allocations = EmptyAllocations();

  ASSERT_OK(executable->ExecuteThunks(allocations, &run_options));
  ASSERT_EQ(executed_streams.size(), 2);
  EXPECT_EQ(executed_streams[0], borrower.last_borrowed_stream());
  EXPECT_EQ(executed_streams[1], borrower.last_borrowed_stream());
  EXPECT_TRUE(lease_was_live_during_execution);
  EXPECT_EQ(borrower.borrow_count(), 1);
  EXPECT_THAT(borrower.borrowed_device_ordinals(),
              UnorderedElementsAre(executor_->device_ordinal()));
  EXPECT_THAT(borrower.borrowed_domains(), UnorderedElementsAre(0));

  ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr returned,
      borrower.pool(/*domain=*/0)->BorrowStream(se::StreamPriority::Default));
  EXPECT_EQ(returned.get(), borrower.last_borrowed_stream());
}

TEST_F(GpuExecutableLocalityDomainTest,
       BackendPoolsSeparateReuseAndInjectGreenContextStreams) {
  if (runtime_locality_domain_count() < 2) {
    GTEST_SKIP() << "Requires at least two runtime locality domains";
  }

  BackendOptions backend_options;
  backend_options.set_platform(platform_);
  backend_options.set_allowed_devices(
      std::set<int>{executor_->device_ordinal()});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Backend> backend,
                       Backend::CreateBackend(backend_options));
  const int device_ordinal = backend->default_device_ordinal();
  se::StreamExecutor* backend_executor = backend->default_stream_executor();
  if (backend_executor->GetDeviceDescription().locality_domain_count() < 2) {
    GTEST_SKIP() << "Backend executor does not expose two locality domains";
  }

  auto borrower = backend->LocalityDomainStreamBorrowerWithPriority();
  absl::StatusOr<StreamPool::Ptr> first_status =
      borrower(device_ordinal, /*domain=*/0, se::StreamPriority::Default);
  if (!first_status.ok()) {
    GTEST_SKIP() << "Locality-domain streams unavailable: "
                 << first_status.status();
  }
  StreamPool::Ptr first = std::move(*first_status);
  se::Stream* first_stream = first.get();
  first.reset();

  ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr reused,
      borrower(device_ordinal, /*domain=*/0, se::StreamPriority::Default));
  EXPECT_EQ(reused.get(), first_stream);
  ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr other_domain,
      borrower(device_ordinal, /*domain=*/1, se::StreamPriority::Default));
  EXPECT_NE(reused.get(), other_domain.get());
  reused.reset();
  other_domain.reset();

  std::vector<se::Stream*> executed_streams(2, nullptr);
  ThunkSequence thunks;
  for (int domain = 0; domain < 2; ++domain) {
    AppendLocalityRegion(thunks, /*thunk_id=*/domain + 1, domain,
                         ThunkSequence::Of<CallbackThunk>(
                             [&, domain](const Thunk::ExecuteParams& params) {
                               executed_streams[domain] = params.stream;
                               return absl::OkStatus();
                             }));
  }

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      CreateExecutable(
          std::move(thunks),
          backend_executor->GetDeviceDescription().locality_domain_count()));
  ASSERT_OK_AND_ASSIGN(StreamPool::Ptr main,
                       backend->BorrowStream(device_ordinal));
  ServiceExecutableRunOptions run_options =
      MakeRunOptions(borrower, backend->memory_allocator(), main.get());
  BufferAllocations allocations = EmptyAllocations(backend->memory_allocator());

  ASSERT_OK(executable->ExecuteThunks(allocations, &run_options));
  ASSERT_NE(executed_streams[0], nullptr);
  ASSERT_NE(executed_streams[1], nullptr);
  EXPECT_NE(executed_streams[0], executed_streams[1]);

  ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr returned0,
      borrower(device_ordinal, /*domain=*/0, se::StreamPriority::Default));
  ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr returned1,
      borrower(device_ordinal, /*domain=*/1, se::StreamPriority::Default));
  EXPECT_EQ(returned0.get(), executed_streams[0]);
  EXPECT_EQ(returned1.get(), executed_streams[1]);

  auto* cuda0 = dynamic_cast<se::gpu::CudaStream*>(executed_streams[0]);
  auto* cuda1 = dynamic_cast<se::gpu::CudaStream*>(executed_streams[1]);
  ASSERT_NE(cuda0, nullptr);
  ASSERT_NE(cuda1, nullptr);
  ASSERT_NE(cuda0->green_context(), nullptr);
  ASSERT_NE(cuda1->green_context(), nullptr);
  EXPECT_NE(cuda0->green_context()->green_ctx(),
            cuda1->green_context()->green_ctx());
}

}  // namespace
}  // namespace xla::gpu
