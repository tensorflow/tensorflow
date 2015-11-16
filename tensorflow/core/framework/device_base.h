#ifndef TENSORFLOW_FRAMEWORK_DEVICE_BASE_H_
#define TENSORFLOW_FRAMEWORK_DEVICE_BASE_H_

#include <memory>
#include <unordered_map>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace Eigen {
class ThreadPoolDevice;
}  // end namespace Eigen

namespace perftools {
namespace gputools {
class Stream;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

class Device;
class Env;
class EventMgr;

namespace thread {
class ThreadPool;
}

// A wrapper for an Eigen Gpu Device that includes per-op state
class PerOpGpuDevice {
 public:
  virtual ~PerOpGpuDevice() {}
  virtual const Eigen::GpuDevice& device() const = 0;
};

// A class that devices can subclass to pass around
// Device-specific context to OpKernels.
class DeviceContext : public core::RefCounted {
 public:
  ~DeviceContext() override {}
  virtual perftools::gputools::Stream* stream() const { return nullptr; }
  virtual void MaintainLifetimeOnStream(
      const Tensor* t, perftools::gputools::Stream* stream) const {}

  // "cpu_tensor" is a tensor on a CPU. Copies "cpu_tensor" into
  // "device_tensor" which is on a GPU device "device". "device_tensor"
  // must be allocated to be of the same size as "cpu_tensor".
  virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                     Tensor* device_tensor,
                                     StatusCallback done) const {
    done(errors::Internal("Unrecognized device type in CPU-to-device Copy"));
  }

  // "device_tensor" is a tensor on a non-CPU device.  Copies
  // device_tensor into "cpu_tensor".  "cpu_tensor" must be allocated
  // to be of the same size as "device_tensor".
  virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                     const string& tensor_name, Device* device,
                                     Tensor* cpu_tensor, StatusCallback done) {
    done(errors::Internal("Unrecognized device type in device-to-CPU Copy"));
  }
};

typedef std::unordered_map<int, DeviceContext*> DeviceContextMap;

class DeviceBase {
 public:
  explicit DeviceBase(Env* env) : env_(env) {}
  virtual ~DeviceBase();

  Env* env() const { return env_; }

  // Override this to return true for devices that require an Op's
  // compute method to save references to the temporary tensors it
  // allocates until the Op execution completes
  virtual bool SaveTemporaryTensors() const { return false; }

  struct CpuWorkerThreads {
    int num_threads = 0;
    thread::ThreadPool* workers = nullptr;
  };

  // Does not take ownership.
  void set_tensorflow_cpu_worker_threads(CpuWorkerThreads* t) {
    cpu_worker_threads_ = t;
  }

  const CpuWorkerThreads* tensorflow_cpu_worker_threads() const {
    CHECK(cpu_worker_threads_ != nullptr);
    return cpu_worker_threads_;
  }

  // "stream" is used in special circumstances (such as the
  // constructors of Ops) where there is no available OpKernelContext.
  // "default_context" is used by OpKernelContext whenever a device does not
  // supply a DeviceContext for an op in FillContextMap (e.g. when only
  // using a single stream.)
  // "event_mgr" is used to delay deallocation of temporary GPU buffers.
  // TODO(pbar) Work out how to move this out of DeviceBase.
  struct GpuDeviceInfo {
    perftools::gputools::Stream* stream;
    DeviceContext* default_context;
    EventMgr* event_mgr;
  };

  // Does not take ownership.
  void set_tensorflow_gpu_device_info(GpuDeviceInfo* g) {
    gpu_device_info_ = g;
  }

  const GpuDeviceInfo* tensorflow_gpu_device_info() const {
    return gpu_device_info_;
  }

  // Does not take ownership.
  void set_eigen_cpu_device(Eigen::ThreadPoolDevice* d) {
    eigen_cpu_device_ = d;
  }

  // Return the Allocator implementation to use based on the allocator
  // attributes requested.  See allocator.h for more details.
  virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
    LOG(FATAL) << "GetAllocator() is not implemented.";
  }

  const Eigen::ThreadPoolDevice* eigen_cpu_device() {
    CHECK(eigen_cpu_device_ != nullptr);
    return eigen_cpu_device_;
  }

  // The caller owns the returned device and must free it by calling
  // DisposeGpuDevice below
  virtual const PerOpGpuDevice* MakeGpuDevice(DeviceContext* /*dc*/,
                                              Allocator* /*allocator*/) {
    // The OpKernelContext calls this even for devices that do not
    // implement an eigen_gpu_device
    return nullptr;
  }

  virtual const DeviceAttributes& attributes() const {
    LOG(FATAL) << "Device does not implement attributes()";
  }

  // Materializes the given TensorProto into 'tensor' stored in Device
  // memory.  Most devices will want to override this.
  //
  // TODO(vrv): We should be able to put this function into
  // OpKernelContext and handle the copies from device memory via send
  // and receive nodes, instead of requiring that each device handle
  // the copies here as well as in copy ops.
  virtual Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                     const AllocatorAttributes alloc_attrs,
                                     Tensor* tensor) {
    return errors::Internal("Device does not implement MakeTensorFromProto()");
  }

 private:
  Env* const env_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  Eigen::ThreadPoolDevice* eigen_cpu_device_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_DEVICE_BASE_H_
