#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
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
  LOG(INFO) << "Local device intra op parallelism threads: "
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
