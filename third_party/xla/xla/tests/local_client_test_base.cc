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
#define EIGEN_USE_THREADS

#include "xla/tests/local_client_test_base.h"

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/client/local_client.h"
#include "xla/client/xla_computation.h"
#include "xla/map_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/test_helpers.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/threadpool.h"

namespace xla {

/* static */ TestAllocator* LocalClientTestBase::allocator_;

absl::StatusOr<se::OwningDeviceMemory> TestAllocator::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  VLOG(2) << "Allocate(" << device_ordinal << ", " << size << ")";
  {
    absl::MutexLock lock(&count_mutex_);
    allocation_count_++;
    device_allocation_count_[device_ordinal]++;
  }
  return se::StreamExecutorMemoryAllocator::Allocate(
      device_ordinal, size, retry_on_failure, memory_space);
}

absl::Status TestAllocator::Deallocate(int device_ordinal,
                                       se::DeviceMemoryBase mem) {
  VLOG(2) << "Deallocate(" << device_ordinal << ")";
  {
    absl::MutexLock lock(&count_mutex_);
    deallocation_count_++;
    device_deallocation_count_[device_ordinal]++;
  }
  return se::StreamExecutorMemoryAllocator::Deallocate(device_ordinal, mem);
}

int64_t TestAllocator::allocation_count() const {
  absl::MutexLock lock(&count_mutex_);
  return allocation_count_;
}

int64_t TestAllocator::allocation_count(int device_ordinal) const {
  absl::MutexLock lock(&count_mutex_);
  auto it = device_allocation_count_.find(device_ordinal);
  if (it == device_allocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

int64_t TestAllocator::deallocation_count() const {
  absl::MutexLock lock(&count_mutex_);
  return deallocation_count_;
}

int64_t TestAllocator::deallocation_count(int device_ordinal) const {
  absl::MutexLock lock(&count_mutex_);
  auto it = device_deallocation_count_.find(device_ordinal);
  if (it == device_deallocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

/* static */ TestAllocator* LocalClientTestBase::GetOrCreateAllocator(
    se::Platform* platform) {
  static absl::Mutex mu(absl::kConstInit);
  absl::MutexLock lock(&mu);

  if (allocator_ == nullptr) {
    allocator_ = new TestAllocator(
        platform == nullptr ? PlatformUtil::GetDefaultPlatform().value()
                            : platform);
  }
  return allocator_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct LocalClientTestBase::EigenThreadPoolWrapper {
  explicit EigenThreadPoolWrapper()
      : pool(new tsl::thread::ThreadPool(tsl::Env::Default(), "XLAEigenTest",
                                         /*num_threads=*/2)),
        device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(),
                                           pool->NumThreads())) {}

  std::unique_ptr<tsl::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

LocalClientTestBase::LocalClientTestBase(se::Platform* platform)
    : local_client_(ClientLibrary::GetOrCreateLocalClient(platform).value()),
      thread_pool_wrapper_(new EigenThreadPoolWrapper()) {
  // Take the first executor, since it's the default one.
  stream_executor_ = PlatformUtil::GetStreamExecutors(local_client_->platform())
                         .value()
                         .front();
  transfer_manager_ =
      TransferManager::GetForPlatform(local_client_->platform()).value();
}

LocalClientTestBase::~LocalClientTestBase() {}

ScopedShapedBuffer LocalClientTestBase::LiteralToShapedBuffer(
    const Literal& literal) {
  return local_client_
      ->LiteralToShapedBuffer(literal, local_client_->default_device_ordinal())
      .value();
}

Literal LocalClientTestBase::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
  return local_client_->ShapedBufferToLiteral(shaped_buffer).value();
}

ExecutableBuildOptions LocalClientTestBase::DefaultExecutableBuildOptions()
    const {
  return ExecutableBuildOptions();
}

ExecutableRunOptions LocalClientTestBase::DefaultExecutableRunOptions() const {
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(thread_pool_wrapper_->device.get());
  run_options.set_allocator(GetOrCreateAllocator(local_client_->platform()));
  return run_options;
}

ScopedShapedBuffer LocalClientTestBase::ExecuteLocallyOrDie(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments) {
  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions())
      .value();
}

ScopedShapedBuffer LocalClientTestBase::ExecuteLocallyOrDie(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
  return ExecuteLocally(computation, arguments, build_options, run_options)
      .value();
}

absl::StatusOr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments) {
  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions());
}

absl::StatusOr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocally(
    const XlaComputation& computation,
    absl::Span<const ShapedBuffer* const> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
  std::vector<const Shape*> argument_layouts(arguments.size());
  for (int i = 0; i < arguments.size(); ++i) {
    argument_layouts[i] = &arguments[i]->on_device_shape();
  }
  TF_ASSIGN_OR_RETURN(
      auto executables,
      local_client_->Compile(computation, argument_layouts, build_options));
  TF_RET_CHECK(executables.size() == 1);
  TF_ASSIGN_OR_RETURN(auto ret, executables[0]->Run(arguments, run_options));

  auto device_ordinal =
      build_options.device_ordinal() == -1 ? 0 : build_options.device_ordinal();
  auto* stream = run_options.stream();
  if (!stream) {
    stream = local_client_->mutable_backend()
                 ->BorrowStream(device_ordinal)
                 .value()
                 .get();
  }
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return std::move(ret);
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
LocalClientTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text) {
  return ParseAndReturnVerifiedModule(hlo_text, HloModuleConfig());
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
LocalClientTestBase::ParseAndReturnVerifiedModule(
    absl::string_view hlo_text, const HloModuleConfig& config) {
  auto module = std::make_unique<VerifiedHloModule>(
      TestName(), config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      local_client_->backend().compiler()->ShapeSizeBytesFunction());
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

}  // namespace xla
