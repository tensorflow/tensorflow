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

#if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM
#error This file must only be included when building with Cuda or ROCm support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/node_file_writer.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/shared_counter.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/tsl/framework/device_id.h"

namespace Eigen {
class StreamInterface;
}

namespace tensorflow {
class GPUKernelTracker;

class ConcretePerOpGpuDevice : public PerOpGpuDevice {
 public:
  ConcretePerOpGpuDevice();

  void Reinitialize(OpKernelContext* context, const void* gpu_stream,
                    tsl::TfDeviceId tf_device_id, Allocator* base_allocator,
                    char* scratch);

  void Reinitialize(OpKernelContext* context, const void* gpu_stream,
                    tsl::PlatformDeviceId platform_device_id,
                    Allocator* base_allocator, char* scratch);

  const Eigen::GpuDevice& device() const override;

 private:
  std::unique_ptr<::Eigen::StreamInterface> stream_device_;
};

class BaseGPUDevice : public LocalDevice {
 public:
  BaseGPUDevice(const SessionOptions& options, const std::string& name,
                Bytes memory_limit, const DeviceLocality& locality,
                tsl::TfDeviceId tf_device_id,
                const std::string& physical_device_desc,
                Allocator* gpu_allocator, Allocator* cpu_allocator,
                bool sync_every_op);

  ~BaseGPUDevice() override;

  // Initialize the device and return the status of initialization.
  Status Init(const SessionOptions& options);

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  Status Sync() override;

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Tensor* output_tensor,
                              const DeviceContext* device_context,
                              StatusCallback done) override;

  // The caller owns the returned device.
  PerOpGpuDevice* MakeGpuDevice() override;

  Status ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                               DeviceContext* dc,
                               Allocator* allocator) override;

  // Returns the platform GPU id of this device within the native driver system;
  // e.g., for CUDA and ROCm this is the ordinal of the GPU within the system.
  int gpu_id() const {
    tsl::PlatformDeviceId platform_device_id;
    TF_CHECK_OK(
        GpuIdManager::TfToPlatformDeviceId(tf_device_id_, &platform_device_id));
    return platform_device_id.value();
  }

  // The executor that provides control for the device; e.g., for CUDA this
  // corresponds to the cuda context.
  se::StreamExecutor* executor() const { return executor_; }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64_t step_id) override;

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return scoped_allocator_mgr_.get();
  }

  // The following two functions always return 0 unless one of the
  // related experimental config options has been specified.

  // If returned value is > 0 then GPU Memory chunks freed before this count
  // are guaranteed not to be in use by any kernel pending on this device.
  uint64 SafeAllocFrontier(uint64 old_value) override;

  // Returns the number of kernels that have been queued for execution on
  // the compute stream and are not yet known to have completed.
  int PendingKernels();

  int priority() const { return stream_->priority; }

  // Helper method for unit tests to reset the streams. Never use in production.
  static void TestOnlyReset();

  void* GetStream() {
    return stream_->compute->implementation()->GpuStreamMemberHack();
  }

 protected:
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned

  se::StreamExecutor* executor_;  // not owned
  std::unique_ptr<ScopedAllocatorMgr> scoped_allocator_mgr_;

 private:
  friend class GPUDeviceTestHelper;
  struct StreamGroup {
    se::Stream* compute = nullptr;
#if TENSORFLOW_USE_ROCM
    se::Stream* nccl = nullptr;
#endif
    se::Stream* host_to_device = nullptr;
    se::Stream* device_to_host = nullptr;
    gtl::InlinedVector<se::Stream*, 4> device_to_device;
    int priority = 0;
  };
  class StreamGroupFactory;

  StreamGroup* stream_;
  mutex scratch_init_mutex_;
  char* scratch_ = nullptr;
  GPUDeviceContext* device_context_;
  DeviceBase::AcceleratorDeviceInfo* accelerator_device_info_ = nullptr;
  mutex trace_mu_;
  tsl::TfDeviceId tf_device_id_;
  const bool sync_every_op_ = false;
  EventMgr* em_ = nullptr;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<GPUKernelTracker> kernel_tracker_;
  int32 pending_cap_ = 0;
  bool timestamped_allocator_ = false;
  NodeFileWriter* node_file_writer_ = nullptr;  // not owned

  // Initialize scratch buffers used by Eigen.
  Status InitScratchBuffers();

  void ReinitializeDevice(OpKernelContext* context, PerOpGpuDevice* device,
                          int stream_id, Allocator* allocator);

  std::string ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                         const int& stream_id);

  // This method returns an initialization status, in addition to
  // calling the "done" StatusCallback, if there is a failure to
  // allocate memory or if the tensor "from" is not DMA-copyable.
  // If there is no error prior to enqueueing the copy, an OK status
  // is returned.
  Status MaybeCopyTensorToGPU(const AllocatorAttributes& alloc_attrs,
                              const Tensor& from, Tensor* to,
                              StatusCallback done);

  Tensor CopyGpuTensorToHostDebugOnly(const Tensor& gpu_tensor);
  void LogInputs(OpKernel* op_kernel, OpKernelContext* context);
  void LogOutputs(OpKernel* op_kernel, OpKernelContext* context);
};

// A per-compute-stream utility that keeps track of kernels that have been
// queued for execution but may not yet have terminated and also the queued
// time of the most recently terminated kernel.
class GPUKernelTracker {
 public:
  // Controls the strategy for inserting tracking events after GPU kernels.
  //   If max_interval >= 0, then insert an event after this many kernels
  //     if an event has not been inserted for another reason.
  //   If max_bytes > 0, then insert an event after kernels allocating this
  //     many bytes have been queued since the last event.
  //   If max_pending > 0, then track up to this many events at once.  If
  //     this limit is reached the GPU::Compute() method will delay starting
  //     additional ops until some event completes.  If 0 and one of the other
  //     fields is non-zero, then a reasonable default will be selected.
  struct Params {
    int max_interval = 0;
    int max_bytes = 0;
    int max_pending = 0;
    Params(int mi, int mb, int mp)
        : max_interval(mi), max_bytes(mb), max_pending(mp) {}
  };

  // If we're going to share a SharedCounter with an allocator, it's owned
  // by the allocator because allocators are initialized once per process.
  // Devices are per-session.
  explicit GPUKernelTracker(const Params& params, Env* env,
                            se::Stream* compute_stream,
                            SharedCounter* timing_counter, Allocator* allocator,
                            EventMgr* event_manager)
      : params_(params),
        env_(env),
        stream_(compute_stream),
        timing_counter_(timing_counter),
        allocator_(allocator),
        em_(event_manager),
        pending_kernels_(
            params.max_pending > 0 ? std::max(8, 2 * params.max_pending) : 64) {
    mem_since_last_ = 0;
    if (!timing_counter_) {
      // There's not a preexisting counter owned by GPUProcessState, i.e.
      // pending_cap > 0 but timestamped_allocator == false.
      owned_counter_ = std::make_unique<SharedCounter>();
      timing_counter_ = owned_counter_.get();
    }
  }

  // Determine whether a GPU kernel should have a recording event queued
  // immediately afterwards.  If so, advance the counter and return the new
  // counter value after enqueuing.
  uint64 MaybeQueue(OpKernelContext* ctx);

  // Record that a GPU kernel has just been enqueued on the compute stream.
  // Inserts the supplied counter value in a new PendingKernel record appended
  // to the end of the ring buffer then returns that same count.
  // Caller is responsible for ensuring that RecordTerminate() is eventually
  // called with the same counter value.
  void RecordQueued(uint64 queued_count, int weight)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Takes a count value returned by RecordQueued and finds the corresponding
  // PendingKernel record in the ring buffer.  Marks the kernel as completed and
  // advances the completion frontier accordingly.
  void RecordTerminated(uint64 queued_count);

  // Returns the largest timing count such that all kernels queued no
  // later than that count are known to have terminated.
  inline uint64 LastTerminatedCount(uint64 old_value) {
    uint64 new_value = last_terminated_count_.load(std::memory_order_relaxed);
    if (new_value == old_value) {
      MaybeQueueProgressEvent();
    }
    return new_value;
  }

  // Returns the number of kernels enqueued that are not yet known to
  // have terminated.
  int NumPending() {
    mutex_lock l(mu_);
    return num_pending_;
  }

  // Yield current thread until number of pending kernels no longer
  // exceeds the cap.
  void PauseWhilePendingExceeds(int cap) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    while (num_pending_ > cap) {
      VLOG(1) << "num_pending_=" << num_pending_ << " cap=" << cap;
      pending_decreased_.wait(l);
    }
  }

 private:
  friend class GPUKernelTrackerTest;
  Params params_;
  Env* env_;
  se::Stream* stream_;
  SharedCounter* timing_counter_;
  std::unique_ptr<SharedCounter> owned_counter_;
  Allocator* allocator_ = nullptr;
  EventMgr* em_ = nullptr;
  std::atomic<uint64> last_terminated_count_ = {1};

  void MaybeQueueProgressEvent();

  // Records when a kernel was queued for execution.  Kernel launches are
  // identified by a unique count value from a per-GPU device timing counter.
  struct PendingKernel {
    uint64 queued_count;
    int weight;
    bool terminated;
    PendingKernel(const PendingKernel& pk) = default;
    PendingKernel() : queued_count(0), weight(0), terminated(false) {}
  };
  mutex mu_;
  int32 mem_since_last_ TF_GUARDED_BY(mu_);
  int32 ops_since_last_ TF_GUARDED_BY(mu_);
  // Ring buffer of PendingKernel records.
  std::vector<PendingKernel> pending_kernels_ TF_GUARDED_BY(mu_);
  // Next unused slot in pending_kernels_.
  int first_available_ TF_GUARDED_BY(mu_) = 0;
  // Last completed PendingKernel such that all prior PendingKernels are
  // also completed.  With out-of-order completion there may be a mixture
  // of completed and uncompleted entries between last_completed_ and
  // first_available_.
  int last_completed_ TF_GUARDED_BY(mu_) = -1;
  // Sum of weights of the outstanding events marking tracked kernels.
  int num_pending_ TF_GUARDED_BY(mu_) = 0;
  condition_variable pending_decreased_ TF_GUARDED_BY(mu_);
};

class BaseGPUDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options,
                       const std::string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
  Status GetDeviceDetails(int device_index,
                          std::unordered_map<string, string>* details) override;

  struct InterconnectMap {
    // Name of interconnect technology, if known.
    std::string name;
    // If possible, strength should approximate Gb/sec bandwidth rate.
    // Where architecture-specific subclassing is not done that won't
    // always be possible.  The minimum expectation is that
    // faster links should have a higher value than slower links.
    int32 strength;
    static const int kSameDeviceStrength;
    static const int kStreamExecutorStrength;
    std::set<std::pair<tsl::PlatformDeviceId, tsl::PlatformDeviceId>>
        directed_links;
  };

 protected:
  // Populates *maps with interconnect maps for all local direct access
  // pathways between GPUs.
  virtual Status GetInterconnectMaps(
      const std::vector<tsl::PlatformDeviceId>& visible_gpu_order,
      se::Platform* gpu_manager, std::vector<InterconnectMap>* maps);

  struct TfDeviceIdHash {
    std::size_t operator()(const tsl::TfDeviceId& id) const noexcept {
      return std::hash<int>{}(id.value());
    }
  };
  typedef std::unordered_map<tsl::TfDeviceId, DeviceLocality, TfDeviceIdHash>
      LocalityMap;
  // Populates *localities with the DeviceLocality descriptor for
  // every TfDeviceId.
  virtual Status GetDeviceLocalities(
      int num_tf_gpus, const std::vector<InterconnectMap>& interconnects,
      LocalityMap* localities);

 private:
  // Creates a BaseGPUDevice associated with 'tf_device_id', allocates
  // (strictly) 'memory_limit' bytes of GPU memory to it, and adds it to the
  // 'devices' vector.
  Status CreateGPUDevice(const SessionOptions& options,
                         const std::string& name_prefix,
                         tsl::TfDeviceId tf_device_id, int64_t memory_limit,
                         const DeviceLocality& dev_locality, size_t num_tf_gpus,
                         std::vector<std::unique_ptr<Device>>* devices);

  virtual std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& dev_locality, tsl::TfDeviceId tf_device_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) = 0;

  Status EnablePeerAccess(
      const std::vector<tsl::PlatformDeviceId>& visible_gpu_order);

  // Returns into 'ids' the list of valid platform GPU ids, in the order that
  // they should map to TF GPU ids "/device:GPU:0", "/device:GPU:1", etc,
  // based upon 'visible_gpu_order' which was generated by parsing
  // GPUOptions::visible_device_list which is a comma-separated list of CUDA or
  // ROCm GPU ids.
  Status GetValidDeviceIds(
      const std::vector<tsl::PlatformDeviceId>& visible_gpu_order,
      std::vector<tsl::PlatformDeviceId>* ids);

  // Cache the valid device IDs if not already cached. Cached IDs are stored in
  // field cached_device_ids_. Passes {0, 1, ..., num_devices-1} to
  // GetValidDeviceIds, so this should only be used in functions where all
  // devices should be treated as visible, like ListPhysicalDevices.
  Status CacheDeviceIds();

  // visible_gpu_initialized_[platform_device_id] is true if visible GPU
  // platform_device_id has been initialized by the process.
  std::unordered_map<int, bool> visible_gpu_initialized_;

  // Cached device IDs, as returned by GetValidDeviceIds when every physical
  // device is visible. Cache should not be used if some devices are not
  // visible.
  std::vector<tsl::PlatformDeviceId> cached_device_ids_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
