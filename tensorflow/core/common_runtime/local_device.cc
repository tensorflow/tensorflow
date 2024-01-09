/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/local_device.h"

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_feature_guard.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

bool OverrideGlobalThreadPoolFromEnvironment() {
  static const bool override_global_threadpool = [] {
    bool flag;
    auto status = ReadBoolFromEnvVar("TF_OVERRIDE_GLOBAL_THREADPOOL",
                                     /*default_val=*/false, &flag);
    if (!status.ok()) {
      LOG(ERROR) << "OverrideGlobalThreadPool: " << status.message();
      return false;
    }
    return flag;
  }();
  return override_global_threadpool;
}

}  // namespace

/* static */
bool LocalDevice::use_global_threadpool_ = true;

struct LocalDevice::EigenThreadPoolInfo {
  // Wrapper so we can provide the CPUAllocator to Eigen for use
  // when ops need extra tmp memory.
  class EigenAllocator : public Eigen::Allocator {
   public:
    explicit EigenAllocator(tensorflow::Allocator* a) : allocator_(a) {}
    void* allocate(size_t num_bytes) const override {
      return allocator_->AllocateRaw(64, num_bytes);
    }
    void deallocate(void* buffer) const override {
      allocator_->DeallocateRaw(buffer);
    }
    tensorflow::Allocator* allocator_;
  };

  explicit EigenThreadPoolInfo(const SessionOptions& options, int numa_node,
                               Allocator* allocator) {
    // Use session setting if specified.
    int32_t intra_op_parallelism_threads =
        options.config.intra_op_parallelism_threads();
    // If no session setting, use environment setting.
    if (intra_op_parallelism_threads == 0) {
      static int env_num_threads = NumIntraOpThreadsFromEnvironment();
      intra_op_parallelism_threads = env_num_threads;
      // If no session setting or environment, compute a reasonable default.
      if (intra_op_parallelism_threads == 0) {
        intra_op_parallelism_threads = port::MaxParallelism(numa_node);
      }
    }
    ThreadOptions thread_opts;
    thread_opts.numa_node = numa_node;
    eigen_worker_threads_.num_threads = intra_op_parallelism_threads;
    eigen_worker_threads_.workers = new thread::ThreadPool(
        options.env, thread_opts, strings::StrCat("numa_", numa_node, "_Eigen"),
        intra_op_parallelism_threads,
        !options.config.experimental().disable_thread_spinning(),
        /*allocator=*/nullptr);
    Eigen::ThreadPoolInterface* threadpool =
        eigen_worker_threads_.workers->AsEigenThreadPool();
    if (allocator != nullptr) {
      eigen_allocator_.reset(new EigenAllocator(allocator));
    }
    eigen_device_.reset(new Eigen::ThreadPoolDevice(
        threadpool, eigen_worker_threads_.num_threads, eigen_allocator_.get()));
  }

  ~EigenThreadPoolInfo() {
    eigen_device_.reset();
    delete eigen_worker_threads_.workers;
  }

  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
  std::unique_ptr<EigenAllocator> eigen_allocator_;
};

LocalDevice::LocalDevice(const SessionOptions& options,
                         const DeviceAttributes& attributes)
    : Device(options.env, attributes), owned_tp_info_(nullptr) {
  // Log info messages if TensorFlow is not compiled with instructions that
  // could speed up performance and are available on the current CPU.
  port::InfoAboutUnusedCPUFeatures();
  LocalDevice::EigenThreadPoolInfo* tp_info;

  if (OverrideGlobalThreadPoolFromEnvironment()) {
    use_global_threadpool_ = false;
  }

  if (use_global_threadpool_) {
    // All ThreadPoolDevices in the process associated with the same
    // NUMA node will share a single fixed sized threadpool for numerical
    // computations.
    static mutex& global_tp_mu = *new mutex;
    static auto& global_tp_info TF_GUARDED_BY(global_tp_mu) =
        *new gtl::InlinedVector<LocalDevice::EigenThreadPoolInfo*, 4>;

    mutex_lock l(global_tp_mu);
    if (options.config.experimental().use_numa_affinity()) {
      int numa_node = attributes.locality().numa_node();
      int num_numa_nodes = port::NUMANumNodes();
      DCHECK_LT(numa_node, num_numa_nodes);
      Allocator* numa_allocator =
          ProcessState::singleton()->GetCPUAllocator(numa_node);
      while (numa_node >= global_tp_info.size()) {
        global_tp_info.push_back(nullptr);
      }
      if (!global_tp_info[numa_node]) {
        global_tp_info[numa_node] = new LocalDevice::EigenThreadPoolInfo(
            options, numa_node, numa_allocator);
      }
      tp_info = global_tp_info[numa_node];
    } else {
      if (global_tp_info.empty()) {
        global_tp_info.push_back(new LocalDevice::EigenThreadPoolInfo(
            options, port::kNUMANoAffinity, nullptr));
      }
      tp_info = global_tp_info[0];
    }
  } else {
    // Each LocalDevice owns a separate ThreadPoolDevice for numerical
    // computations.
    // TODO(tucker): NUMA for these too?
    owned_tp_info_.reset(new LocalDevice::EigenThreadPoolInfo(
        options, port::kNUMANoAffinity, nullptr));
    tp_info = owned_tp_info_.get();
  }

  VLOG(1) << "LocalDevice using CPU work thread pool: "
          << tp_info->eigen_worker_threads_.workers
          << ", num_threads=" << tp_info->eigen_worker_threads_.num_threads;

  set_tensorflow_cpu_worker_threads(&tp_info->eigen_worker_threads_);
  set_eigen_cpu_device(tp_info->eigen_device_.get());
}

LocalDevice::~LocalDevice() {}

}  // namespace tensorflow
