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
#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/tests/local_client_test_base.h"

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ TestAllocator* LocalClientTestBase::allocator_;

StatusOr<perftools::gputools::DeviceMemoryBase> TestAllocator::Allocate(
    int device_ordinal, uint64 size, bool retry_on_failure) {
  VLOG(2) << "Allocate(" << device_ordinal << ", " << size << ")";
  {
    tensorflow::mutex_lock lock(count_mutex_);
    allocation_count_++;
    device_allocation_count_[device_ordinal]++;
  }
  return StreamExecutorMemoryAllocator::Allocate(device_ordinal, size);
}

tensorflow::Status TestAllocator::Deallocate(
    int device_ordinal, perftools::gputools::DeviceMemoryBase* mem) {
  VLOG(2) << "Deallocate(" << device_ordinal << ")";
  {
    tensorflow::mutex_lock lock(count_mutex_);
    deallocation_count_++;
    device_deallocation_count_[device_ordinal]++;
  }
  return StreamExecutorMemoryAllocator::Deallocate(device_ordinal, mem);
}

int64 TestAllocator::allocation_count() const {
  tensorflow::mutex_lock lock(count_mutex_);
  return allocation_count_;
}

int64 TestAllocator::allocation_count(int device_ordinal) const {
  tensorflow::mutex_lock lock(count_mutex_);
  auto it = device_allocation_count_.find(device_ordinal);
  if (it == device_allocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

int64 TestAllocator::deallocation_count() const {
  tensorflow::mutex_lock lock(count_mutex_);
  return deallocation_count_;
}

int64 TestAllocator::deallocation_count(int device_ordinal) const {
  tensorflow::mutex_lock lock(count_mutex_);
  auto it = device_deallocation_count_.find(device_ordinal);
  if (it == device_deallocation_count_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

/* static */ TestAllocator* LocalClientTestBase::GetOrCreateAllocator(
    perftools::gputools::Platform* platform) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  tensorflow::mutex_lock lock(mu);

  if (allocator_ == nullptr) {
    allocator_ = new TestAllocator(
        platform == nullptr ? PlatformUtil::GetDefaultPlatform().ValueOrDie()
                            : platform);
  }
  return allocator_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct LocalClientTestBase::EigenThreadPoolWrapper {
  explicit EigenThreadPoolWrapper()
      : pool(new tensorflow::thread::ThreadPool(
            tensorflow::Env::Default(), "XLAEigenTest", /*num_threads=*/2)),
        wrapper(new tensorflow::EigenThreadPoolWrapper(pool.get())),
        device(new Eigen::ThreadPoolDevice(wrapper.get(),
                                           wrapper->NumThreads())) {}

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  std::unique_ptr<tensorflow::EigenThreadPoolWrapper> wrapper;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

LocalClientTestBase::LocalClientTestBase(
    perftools::gputools::Platform* platform)
    : local_client_(
          ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie()),
      thread_pool_wrapper_(new EigenThreadPoolWrapper()) {
  stream_executor_ = PlatformUtil::GetStreamExecutors(local_client_->platform())
                         .ValueOrDie()[local_client_->default_device_ordinal()];
  transfer_manager_ =
      TransferManager::GetForPlatform(local_client_->platform()).ValueOrDie();
}

LocalClientTestBase::~LocalClientTestBase() {}

std::unique_ptr<ScopedShapedBuffer> LocalClientTestBase::LiteralToShapedBuffer(
    const Literal& literal) {
  return local_client_
      ->LiteralToShapedBuffer(literal, local_client_->default_device_ordinal())
      .ConsumeValueOrDie();
}

std::unique_ptr<Literal> LocalClientTestBase::ShapedBufferToLiteral(
    const ShapedBuffer& shaped_buffer) {
  return local_client_->ShapedBufferToLiteral(shaped_buffer)
      .ConsumeValueOrDie();
}

ExecutableBuildOptions LocalClientTestBase::DefaultExecutableBuildOptions()
    const {
  return ExecutableBuildOptions();
}

ExecutableRunOptions LocalClientTestBase::DefaultExecutableRunOptions() const {
  ExecutableRunOptions run_options;
  run_options.set_inter_op_thread_pool(
      local_client_->backend().inter_op_thread_pool());
  run_options.set_intra_op_thread_pool(thread_pool_wrapper_->device.get());
  run_options.set_allocator(GetOrCreateAllocator(local_client_->platform()));
  return run_options;
}

std::unique_ptr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocallyOrDie(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions())
      .ConsumeValueOrDie();
}

std::unique_ptr<ScopedShapedBuffer> LocalClientTestBase::ExecuteLocallyOrDie(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
  return ExecuteLocally(computation, arguments, build_options, run_options)
      .ConsumeValueOrDie();
}

StatusOr<std::unique_ptr<ScopedShapedBuffer>>
LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments) {
  return ExecuteLocally(computation, arguments, DefaultExecutableBuildOptions(),
                        DefaultExecutableRunOptions());
}

StatusOr<std::unique_ptr<ScopedShapedBuffer>>
LocalClientTestBase::ExecuteLocally(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
    const ExecutableBuildOptions& build_options,
    const ExecutableRunOptions& run_options) {
  std::vector<const Shape*> argument_layouts(arguments.size());
  for (int i = 0; i < arguments.size(); ++i) {
    argument_layouts[i] = &arguments[i]->shape();
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> executable,
      local_client_->Compile(computation, argument_layouts, build_options));
  return executable->Run(arguments, run_options);
}

}  // namespace xla
