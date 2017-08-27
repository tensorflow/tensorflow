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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_feature_guard.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

/* static */
bool LocalDevice::use_global_threadpool_ = true;

struct LocalDevice::EigenThreadPoolInfo {
  explicit EigenThreadPoolInfo(const SessionOptions& options) {
    int32 intra_op_parallelism_threads =
        options.config.intra_op_parallelism_threads();
    if (intra_op_parallelism_threads == 0) {
      intra_op_parallelism_threads = port::NumSchedulableCPUs();
    }
    VLOG(1) << "Local device intra op parallelism threads: "
            << intra_op_parallelism_threads;
    eigen_worker_threads_.num_threads = intra_op_parallelism_threads;
    eigen_worker_threads_.workers = new thread::ThreadPool(
        options.env, "Eigen", intra_op_parallelism_threads);
    eigen_threadpool_wrapper_.reset(
        new EigenThreadPoolWrapper(eigen_worker_threads_.workers));
    eigen_device_.reset(new Eigen::ThreadPoolDevice(
        eigen_threadpool_wrapper_.get(), eigen_worker_threads_.num_threads));
  }

  ~EigenThreadPoolInfo() {
    eigen_threadpool_wrapper_.reset();
    eigen_device_.reset();
    delete eigen_worker_threads_.workers;
  }

  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolInterface> eigen_threadpool_wrapper_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
};

LocalDevice::LocalDevice(const SessionOptions& options,
                         const DeviceAttributes& attributes)
    : Device(options.env, attributes), owned_tp_info_(nullptr) {
  // Log info messages if TensorFlow is not compiled with instructions that
  // could speed up performance and are available on the current CPU.
  port::InfoAboutUnusedCPUFeatures();
  LocalDevice::EigenThreadPoolInfo* tp_info;
  if (use_global_threadpool_) {
    // All ThreadPoolDevices in the process will use this single fixed
    // sized threadpool for numerical computations.
    static LocalDevice::EigenThreadPoolInfo* global_tp_info =
        new LocalDevice::EigenThreadPoolInfo(options);
    tp_info = global_tp_info;
  } else {
    // Each LocalDevice owns a separate ThreadPoolDevice for numerical
    // computations.
    owned_tp_info_.reset(new LocalDevice::EigenThreadPoolInfo(options));
    tp_info = owned_tp_info_.get();
  }
  set_tensorflow_cpu_worker_threads(&tp_info->eigen_worker_threads_);
  set_eigen_cpu_device(tp_info->eigen_device_.get());
}

LocalDevice::~LocalDevice() {}

}  // namespace tensorflow
