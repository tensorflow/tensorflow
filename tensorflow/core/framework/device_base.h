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

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/macros.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"

namespace Eigen {
struct ThreadPoolDevice;
#ifdef TENSORFLOW_USE_SYCL
struct SyclDevice;
#endif
}  // end namespace Eigen

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {

class Device;
class DeviceAttributes;
class Env;
class EventMgr;
class OpKernelContext;
class ResourceMgr;
class ScopedAllocatorMgr;
class TensorProto;

namespace thread {
class ThreadPool;
}

// A wrapper for an Eigen Gpu Device that includes per-op state. The
// class is defined even for non-GPU devices since the
// OpKernelContext::Params structure wants to fill it in.
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
  virtual stream_executor::Stream* stream() const { return nullptr; }
  virtual void MaintainLifetimeOnStream(const Tensor* t,
                                        stream_executor::Stream* stream) const {
  }

  // "cpu_tensor" is a tensor on a CPU. Copies "cpu_tensor" into
  // "device_tensor" which is on a non-CPU device "device". "device_tensor"
  // must be allocated to be of the same size as "cpu_tensor".
  virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                     Tensor* device_tensor, StatusCallback done,
                                     bool sync_dst_compute = true) const {
    done(errors::Internal("Unrecognized device type in CPU-to-device Copy"));
  }

  // Copies a tensor in this device.
  virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                                      Device* device, Tensor* output_tensor,
                                      StatusCallback done) const {
    done(errors::Unimplemented("Copy in same device not implemented."));
  }

  // "device_tensor" is a tensor on a non-CPU device.  Copies
  // device_tensor into "cpu_tensor".  "cpu_tensor" must be allocated
  // to be of the same size as "device_tensor".
  virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                     StringPiece tensor_name, Device* device,
                                     Tensor* cpu_tensor, StatusCallback done) {
    done(errors::Internal("Unrecognized device type in device-to-CPU Copy"));
  }

  // If possible, wait for all events on *stream to complete then execute func.
  // A non-OK Status is returned otherwise.  The stream argument should be the
  // one provided by GpuDeviceInfo.  This function is not applicable to devices
  // that don't provide such a value.
  virtual Status ThenExecute(Device* device, stream_executor::Stream* stream,
                             std::function<void()> func) {
    return errors::Internal("ThenExecute not supported by device");
  }
};

// map[i] is the DeviceContext* for the node with id i, if i < map.size().
typedef std::vector<DeviceContext*> DeviceContextMap;

class DeviceBase {
 public:
  explicit DeviceBase(Env* env) : env_(env) {}
  virtual ~DeviceBase();

  Env* env() const { return env_; }

  // Override this to return true for devices that require an Op's
  // compute method to save references to the temporary tensors it
  // allocates until the Op execution completes
  virtual bool RequiresRecordingAccessedTensors() const { return false; }

  struct CpuWorkerThreads {
    int num_threads = 0;
    thread::ThreadPool* workers = nullptr;
  };

  // Does not take ownership.
  void set_tensorflow_cpu_worker_threads(CpuWorkerThreads* t) {
    cpu_worker_threads_ = t;
  }

  virtual const CpuWorkerThreads* tensorflow_cpu_worker_threads() const {
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
  // GpuDeviceInfo name is an unfortunate legacy, it is used not only by GPUs
  // but also by TPU devices (to provide default device context).
  struct GpuDeviceInfo {
    // Make sure all the defaults are NULL, so we can spot missing assignments.
    stream_executor::Stream* stream = nullptr;
    DeviceContext* default_context = nullptr;
    EventMgr* event_mgr = nullptr;
    int gpu_id = -1;
  };

  // Does not take ownership.
  void set_tensorflow_gpu_device_info(GpuDeviceInfo* g) {
    gpu_device_info_ = g;
  }

  virtual const GpuDeviceInfo* tensorflow_gpu_device_info() const {
    return gpu_device_info_;
  }

  // The preferred thread pool for this device. If it is nullptr, the system
  // automatically assigns a thread pool for execution.
  virtual thread::ThreadPool* tensorflow_device_thread_pool() {
    return device_thread_pool_;
  }

  // Does not take ownership.
  void set_eigen_cpu_device(Eigen::ThreadPoolDevice* d);

#ifdef TENSORFLOW_USE_SYCL
  void set_eigen_sycl_device(Eigen::SyclDevice* d) { eigen_sycl_device_ = d; }
#endif

  // Return the Allocator implementation to use based on the allocator
  // attributes requested.  See allocator.h for more details.
  virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
    LOG(FATAL) << "GetAllocator() is not implemented.";
    return nullptr;
  }

  // This method is provided for backwards compatibility, and will be removed
  // in a future release.
  ABSL_DEPRECATED("Use `this->GetAllocator()` or `this->GetScopedAllocator()`.")
  Allocator* GetStepAllocator(AllocatorAttributes attr, ResourceMgr*) {
    return GetAllocator(attr);
  }

  // Return an Allocator prepared for use in particular places by graph
  // optimization
  virtual Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                        int64 step_id) {
    LOG(FATAL) << "Device does not implement GetScopedAllocator()";
    return nullptr;
  }

  virtual ScopedAllocatorMgr* GetScopedAllocatorMgr() const { return nullptr; }

  virtual bool has_eigen_cpu_device() const {
    return !eigen_cpu_devices_.empty();
  }

  virtual const Eigen::ThreadPoolDevice* eigen_cpu_device();

#ifdef TENSORFLOW_USE_SYCL
  virtual const Eigen::SyclDevice* eigen_sycl_device() const {
    CHECK(eigen_sycl_device_ != nullptr);
    return eigen_sycl_device_;
  }
#endif

  // Caller owns the return value. The OpKernelContext calls this even
  // for devices that do not implement an eigen_gpu_device. Overridden
  // by GPU devices to return a derived type.
  virtual PerOpGpuDevice* MakeGpuDevice() { return nullptr; }

  virtual DeviceBase* UnderlyingDevice() { return this; }
  virtual const DeviceBase* UnderlyingDevice() const { return this; }

  // This is overridden by GPU devices to reinitialize the derived
  // type returned by MakeGpuDevice.
  virtual Status ReinitializeGpuDevice(OpKernelContext* /*context*/,
                                       PerOpGpuDevice* /*device*/,
                                       DeviceContext* /*dc*/,
                                       Allocator* /*allocator*/) {
    return Status::OK();
  }

  // Unimplemented by default
  virtual const DeviceAttributes& attributes() const;
  virtual const string& name() const;

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

  // Some devices (i.e. GPUs) may free device memory prior to its actual use
  // being completed on the assumption that subsequent allocations can only be
  // used serially with respect to pending uses.  If this function returns a
  // non-zero value it is the value of a device-specific counter such that any
  // device memory tagged with an earlier freed-at count is really unencumbered
  // by pending uses.  For this to be useful the device memory allocator must
  // be tagging deallocated memory chunks using the same counter.
  virtual uint64 SafeAllocFrontier(uint64 old_value) { return 0; }

  // Copies `input_tensor` to `output_tensor`, where both tensors are on this
  // device. This function assumes that `output_tensor` has already been
  // allocated with a buffer that is large enough to hold `input_tensor`'s data.
  // Calls `done` from a device-specific thread after copy is finished, which
  // may be the same as calling thread.
  //
  // NOTE(ayushd): This function is for TensorFlow internal use only.  Deep copy
  // is discouraged and should not be used in OpKernels.
  virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
                                      Tensor* output_tensor,
                                      const DeviceContext* device_context,
                                      StatusCallback done) {
    done(errors::Internal("Device ", name(), " does not implement ",
                          "CopyTensorInSameDevice"));
  }

 protected:
  // Does not take ownership.
  void set_tensorflow_device_thread_pool(thread::ThreadPool* thread_pool) {
    device_thread_pool_ = thread_pool;
  }

 private:
  Env* const env_;
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;
  // Set by GPUs as well as by TPU devices.
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  thread::ThreadPool* device_thread_pool_ = nullptr;
  std::vector<Eigen::ThreadPoolDevice*> eigen_cpu_devices_;
#ifdef TENSORFLOW_USE_SYCL
  Eigen::SyclDevice* eigen_sycl_device_ = nullptr;
#endif
};

// Methods to create and check for Symbolic execution devices.
// Such devices are mostly used for TF-XLA bridge. TF should not treat these as
// normal devices.
void AddSymbolicExecutionDevice(absl::string_view device_name);
bool IsSymbolicExecutionDevice(absl::string_view device_name);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_BASE_H_
