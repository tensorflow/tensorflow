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
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

DeviceBase::CpuWorkerThreads eigen_worker_threads;
Eigen::ThreadPoolInterface* eigen_thread_pool = nullptr;
Eigen::ThreadPoolDevice* eigen_device = nullptr;

static bool InitModule(const SessionOptions& options) {
  int32 intra_op_parallelism_threads =
      options.config.intra_op_parallelism_threads();
  if (intra_op_parallelism_threads == 0) {
    intra_op_parallelism_threads = port::NumSchedulableCPUs();
  }
  VLOG(1) << "Local device intra op parallelism threads: "
          << intra_op_parallelism_threads;
  eigen_worker_threads.num_threads = intra_op_parallelism_threads;
  eigen_worker_threads.workers = new thread::ThreadPool(
      options.env, "Eigen", intra_op_parallelism_threads);
  eigen_thread_pool = new EigenThreadPoolWrapper(eigen_worker_threads.workers);
  eigen_device = new Eigen::ThreadPoolDevice(eigen_thread_pool,
                                             eigen_worker_threads.num_threads);
  return true;
}
}  // end namespace

// LocalDevice ----------------------------------------------------------------

LocalDevice::LocalDevice(const SessionOptions& options,
                         const DeviceAttributes& attributes,
                         Allocator* device_allocator)
    : Device(options.env, attributes, device_allocator) {
  // All ThreadPoolDevices in the process will use this single fixed
  // sized threadpool for numerical computations.
  static bool init = InitModule(options);
  CHECK(init);  // Avoids compiler warning that init is unused.
  set_tensorflow_cpu_worker_threads(&eigen_worker_threads);
  set_eigen_cpu_device(eigen_device);
}

}  // namespace tensorflow
