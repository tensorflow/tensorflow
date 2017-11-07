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

// TODO(opensource): Use a more generic sounding preprocessor name than
// GOOGLE_CUDA
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <list>
#include <map>
#include <tuple>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_stream_util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// Eigen Ops directly allocate memory only for temporary buffers used
// during OpKernel::Compute().  The recommended way of allocating such
// memory is via OpKernelContext::allocate_temp().  However, Eigen Ops
// don't have access to OpKernelContext, instead they get access to
// memory directly through the device allocator.  As an Open Source
// project, Eigen assumes allocator semantics similar to those of the
// CUDA memory allocator, and may not work correctly due to race
// conditions if used with some other allocator.  For safety, we need
// to delay deallocation calls out of Eigen until all events on the
// corresponding stream have completed.  The following two classes
// serve this purpose in two different compilation environments.

class EigenCudaStreamDevice : public ::Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice()
      : scratch_(nullptr), semaphore_(nullptr), context_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}
  void Reinitialize(OpKernelContext* context, const cudaStream_t* cuda_stream,
                    int gpu_id, ::tensorflow::Allocator* alloc, char* scratch) {
    if (LogMemory::IsEnabled()) {
      operation_ = context->op_kernel().name() + "/EigenAllocator";
      step_id_ = context->step_id();
    }
    context_ = context;
    scratch_ = scratch;
    semaphore_ =
        reinterpret_cast<unsigned int*>(scratch + Eigen::kCudaScratchSize);
    stream_ = cuda_stream;
    allocator_ = alloc;
    device_prop_ = &Eigen::m_deviceProperties[gpu_id];
  }

  const cudaStream_t& stream() const override { return *stream_; }
  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    void* ret = allocator_->AllocateRaw(32 /* alignment */, num_bytes);
    if (ret == nullptr) {
      if (context_) {
        context_->SetStatus(errors::ResourceExhausted(
            strings::StrCat("Ran out of GPU memory when allocating ", num_bytes,
                            " bytes for ", operation_)));
      } else {
        LOG(FATAL)
            << "EigenAllocator for GPU ran out of memory when allocating "
            << num_bytes << ". See error logs for more detailed info.";
      }
    }
    if (LogMemory::IsEnabled() && ret != nullptr) {
      LogMemory::RecordRawAllocation(operation_, step_id_, num_bytes, ret,
                                     allocator_);
    }
    return ret;
  }
  void deallocate(void* buffer) const override {
    if (LogMemory::IsEnabled() && buffer != nullptr) {
      LogMemory::RecordRawDeallocation(operation_, step_id_, buffer, allocator_,
                                       true);
    }
    AsyncFreeData* afData =
        new AsyncFreeData(allocator_, buffer, operation_, step_id_);
    cudaError_t err = cudaStreamAddCallback(*stream_, asyncFree, afData, 0);
    CHECK_EQ(err, cudaSuccess);
  }

  // Return a pointer to a per stream scratchpad of 1024 bytes residing
  // in global memory.
  void* scratchpad() const override { return scratch_; }

  // Return a semaphore. The semaphore is initially initialized to 0, and
  // each kernel using it is responsible for resetting to 0 upon completion
  // to maintain the invariant that the semaphore is always equal to 0 upon
  // each kernel start.
  unsigned int* semaphore() const override { return semaphore_; }

 private:
  struct AsyncFreeData {
    AsyncFreeData(::tensorflow::Allocator* a, void* p, const string& o,
                  const int64 s)
        : allocator_(a), address_(p), operation_(o), step_id_(s) {}
    ::tensorflow::Allocator* allocator_;
    void* address_;
    const string operation_;
    const int64 step_id_;
  };

  static void CUDART_CB asyncFree(cudaStream_t stream, cudaError_t status,
                                  void* userData) {
    AsyncFreeData* data = static_cast<AsyncFreeData*>(userData);
    if (LogMemory::IsEnabled()) {
      LogMemory::RecordRawDeallocation(data->operation_, data->step_id_,
                                       data->address_, data->allocator_, false);
    }
    data->allocator_->DeallocateRaw(data->address_);
    delete data;
  }

  string operation_;
  int64 step_id_;
  const cudaStream_t* stream_;          // Not owned.
  const cudaDeviceProp* device_prop_;   // Not owned.
  ::tensorflow::Allocator* allocator_;  // Not owned.
  mutable char* scratch_;
  mutable unsigned int* semaphore_;
  OpKernelContext* context_;

  TF_DISALLOW_COPY_AND_ASSIGN(EigenCudaStreamDevice);
};

// This factory helps to ensure that different GPU device objects that refer to
// the same physical device and stream group id use the same stream group
// object (and therefore the same CUDA streams). This is necessary since there
// is a single memory allocator per device (see ProcessState::GetGPUAllocator)
// and allocators must not be shared across streams.
class BaseGPUDevice::StreamGroupFactory {
 public:
  // Returns the unique stream group for use with the stream defined by
  // {gpu_id, stream_group_within_gpu}, creating it if it does not yet exist.
  // This function is thread safe.
  BaseGPUDevice::StreamGroup* GetOrCreate(int gpu_id,
                                          int stream_group_within_gpu,
                                          gpu::StreamExecutor* executor) {
    mutex_lock guard(lock_);
    StreamGroup* group = &streams_[key_type(gpu_id, stream_group_within_gpu)];
    if (!group->compute) {
      group->compute = new gpu::Stream(executor);
      group->compute->Init();
      VLOG(2) << "Created stream[" << stream_group_within_gpu
              << "] = " << group->compute;

      group->host_to_device = new gpu::Stream(executor);
      group->host_to_device->Init();
      VLOG(2) << "Created host_to_device_stream[" << stream_group_within_gpu
              << "] = " << group->host_to_device;

      group->device_to_host = new gpu::Stream(executor);
      group->device_to_host->Init();
      VLOG(2) << "Created device_to_host_stream[" << stream_group_within_gpu
              << "] = " << group->device_to_host;

      group->device_to_device = new gpu::Stream(executor);
      group->device_to_device->Init();
      VLOG(2) << "Created device_to_device_stream[" << stream_group_within_gpu
              << "] = " << group->device_to_host;
    }
    return group;
  }

  // Returns a reference to the StreamGroupFactory singleton. Note that this is
  // never destroyed, so the objects it owns are never deleted.
  static StreamGroupFactory& Global() {
    static StreamGroupFactory* instance = new StreamGroupFactory();
    return *instance;
  }

 private:
  mutex lock_;
  using key_type = std::tuple<int, int>;
  std::map<key_type, StreamGroup> streams_;

  // StreamGroupFactory cannot be created directly; Call
  // StreamGroupFactory::Global() to get the global instance.
  StreamGroupFactory() = default;
  TF_DISALLOW_COPY_AND_ASSIGN(StreamGroupFactory);
};

BaseGPUDevice::BaseGPUDevice(const SessionOptions& options, const string& name,
                             Bytes memory_limit, const DeviceLocality& locality,
                             int gpu_id, const string& physical_device_desc,
                             Allocator* gpu_allocator, Allocator* cpu_allocator,
                             bool sync_every_op, int32 max_streams)
    : LocalDevice(options, Device::BuildDeviceAttributes(name, DEVICE_GPU,
                                                         memory_limit, locality,
                                                         physical_device_desc)),
      gpu_allocator_(gpu_allocator),
      cpu_allocator_(cpu_allocator),
      gpu_id_(gpu_id),
      sync_every_op_(sync_every_op),
      max_streams_(max_streams) {
  ProcessState::singleton()->EnableGPUDevice();
}

BaseGPUDevice::~BaseGPUDevice() {
  delete gpu_device_info_;
  for (auto ctx : device_contexts_) ctx->Unref();
}

Status BaseGPUDevice::Init(const SessionOptions& options) {
  auto executor_status = GPUMachineManager()->ExecutorForDevice(gpu_id_);
  if (!executor_status.status().ok()) {
    return errors::Internal("Failed to get StreamExecutor for device ",
                            gpu_id_);
  }

  executor_ = executor_status.ValueOrDie();
  em_.reset(new EventMgr(executor_, options.config.gpu_options()));

  if (max_streams_ < 1) {
    return errors::InvalidArgument("Invalid value for max_streams.");
  }

  // Create the specified number of GPU streams
  for (int i = 0; i < max_streams_; i++) {
    streams_.push_back(
        StreamGroupFactory::Global().GetOrCreate(gpu_id_, i, executor_));

    size_t scratch_buffer_size = Eigen::kCudaScratchSize + sizeof(unsigned int);
    void* scratch_buffer = gpu_allocator_->AllocateRaw(
        Allocator::kAllocatorAlignment, scratch_buffer_size);
    if (scratch_buffer == nullptr) {
      return errors::FailedPrecondition(
          "Failed to allocate scratch buffer for device ", gpu_id_);
    }
    scratch_.push_back(static_cast<char*>(scratch_buffer));

    perftools::gputools::DeviceMemory<char> mem(
        perftools::gputools::DeviceMemoryBase(scratch_buffer,
                                              scratch_buffer_size));

    bool ok = executor_->SynchronousMemZero(
        &mem, Eigen::kCudaScratchSize + sizeof(unsigned int));
    if (!ok) {
      return errors::FailedPrecondition(
          "Failed to memcopy into scratch buffer for device ", gpu_id_);
    }

    device_contexts_.push_back(new GPUDeviceContext(
        i, streams_.back()->compute, streams_.back()->host_to_device,
        streams_.back()->device_to_host, streams_.back()->device_to_device));
  }
  gpu_device_info_ = new GpuDeviceInfo;
  gpu_device_info_->stream = streams_[0]->compute;
  gpu_device_info_->default_context = device_contexts_[0];
  gpu_device_info_->event_mgr = em_.get();
  gpu_device_info_->gpu_id = gpu_id_;
  set_tensorflow_gpu_device_info(gpu_device_info_);

  return Status::OK();
}

bool BaseGPUDevice::RequiresRecordingAccessedTensors() const {
  // When there is no more than one stream, we release the tensor reference
  // at the end of the kernel launch, instead of at the end of the kernel
  // execution.
  return streams_.size() > 1;
}

Status BaseGPUDevice::FillContextMap(const Graph* graph,
                                     DeviceContextMap* device_context_map) {
  VLOG(2) << "FillContextMap";

  const size_t num_streams = streams_.size();
  // Special case for single stream.
  if (num_streams == 1) {
    return Status::OK();
  }
  const int64 before = Env::Default()->NowMicros();
  gpu_stream_util::AssignStreamsOpts opts;
  opts.max_streams = static_cast<int32>(num_streams);
  std::unordered_map<int, int> node_to_stream_id;
  TF_RETURN_IF_ERROR(
      gpu_stream_util::AssignStreams(graph, opts, &node_to_stream_id));
  int64 elapsed = Env::Default()->NowMicros() - before;
  VLOG(3) << "AssignStreams took " << elapsed << "us";

  // Fill in the context map.  It is OK for this map to contain
  // duplicate DeviceContexts so long as we increment the refcount.
  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    auto mapped_stream = node_to_stream_id[n->id()];
    CHECK_LE(mapped_stream, num_streams);
    auto ctx = device_contexts_[mapped_stream];
    VLOG(3) << "Assigned stream " << node_to_stream_id[n->id()]
            << " ==> stream[" << ctx->stream_id() << "] for node id " << n->id()
            << " " << n->type_string() << " " << n->name();
    ctx->Ref();
    (*device_context_map)[n->id()] = ctx;
  }

  return Status::OK();
}

void BaseGPUDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  // ScopedActivity is cheap when tracing is not active, but we
  // can avoid computing the Hash64.
  // TODO(pbar) This would no longer be needed if Ops have a unique id.
  const uint64 id = port::Tracing::IsActive() ? Hash64(op_kernel->name()) : 0;
  port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                       id);

  // NOTE(tucker): We need to discriminate between Eigen GPU
  // operations and all others.  If an operation is Eigen
  // implemented (or otherwise tries to launch a cuda kernel
  // directly), we need to establish a stacked-scoped environment
  // that directs it to execute on the proper device.  Otherwise we
  // expect the Op to use StreamExecutor directly and correctly.  The
  // way we make this discrimination is quite hacky: At the moment
  // the only non-Eigen GPU Op is the recv-op, which is known to be
  // asynchronous.
  if (op_kernel->is_internal() && op_kernel->type_string() == "_Recv") {
    context->SetStatus(errors::Internal(
        "Invalid synchronous 'Compute' on GPU for '_Recv' op"));
  } else if (port::Tracing::ScopedAnnotation::Enabled()) {
    port::Tracing::ScopedAnnotation annotation(op_kernel->name(),
                                               op_kernel->type_string());
    ComputeHelper(op_kernel, context);
  } else {
    ComputeHelper(op_kernel, context);
  }
}

void BaseGPUDevice::ComputeHelper(OpKernel* op_kernel,
                                  OpKernelContext* context) {
  GPUDeviceContext* gpu_device_context = device_contexts_[0];
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  gpu::Stream* stream = gpu_device_context->stream();
  const auto stream_id = gpu_device_context->stream_id();

  const bool vlog_1 = VLOG_IS_ON(1);
  const bool vlog_2 = vlog_1 && VLOG_IS_ON(2);

  if (vlog_1) {
    VLOG(1) << "GpuDevice::Compute " << op_kernel->name() << " op "
            << op_kernel->type_string() << " on GPU" << gpu_id_ << " stream["
            << stream_id << "]";
  }

  const auto num_streams = streams_.size();
  if (num_streams > 1) {
    // If this op's device context is different from the other contexts,
    // we must wait on the stream.
    for (int i = 0; i < context->num_inputs(); ++i) {
      const GPUDeviceContext* idc =
          static_cast<GPUDeviceContext*>(context->input_device_context(i));
      OP_REQUIRES(context, idc != nullptr,
                  errors::Internal("Input device context ", i,
                                   " was not set properly."));
      if (vlog_2) {
        const void* base;
        size_t len;
        if (context->has_input(i)) {
          if (IsRefType(context->input_dtype(i))) {
            Tensor tensor = context->mutable_input(i, false);
            base = DMAHelper::base(&tensor);
            len = tensor.TotalBytes();
          } else {
            const Tensor& tensor = context->input(i);
            base = DMAHelper::base(&tensor);
            len = tensor.TotalBytes();
          }
          LOG(INFO) << "Input " << i << " " << base << "  " << len;
          LOG(INFO) << "  stream[" << stream_id << "].ThenWaitFor(stream["
                    << idc->stream_id() << "])"
                    << ((idc->stream() == stream) ? " not needed" : "");
        }
      }
      if (idc->stream() != stream) stream->ThenWaitFor(idc->stream());
    }
  }
  gpu::cuda::ScopedActivateExecutorContext scoped_activation{stream->parent()};
  op_kernel->Compute(context);
  if (context->status().ok()) {
    if (sync_every_op_) {
      // Note: GPUUtil::Sync() only syncs the default stream.
      // We need to either sync the stream used by this op, or
      // all streams.  Given that this flag is typically used for
      // debugging it makes more sense to sync all GPU activity.
      context->SetStatus(GPUUtil::SyncAll(this));
    }
  }
}

void BaseGPUDevice::ConsumeListOfAccessedTensors(
    DeviceContext* device_context, const TensorReferenceVector& tensor_refs) {
  GPUDeviceContext* gpu_device_context = device_contexts_[0];
  if (device_context != nullptr) {
    gpu_device_context = static_cast<GPUDeviceContext*>(device_context);
  }
  gpu::Stream* stream = gpu_device_context->stream();
  em_->ThenDeleteTensors(stream, tensor_refs);
}

// Based on the semantics of Device::Sync this call should wait for
// all streams not just the current one.
Status BaseGPUDevice::Sync() { return GPUUtil::SyncAll(this); }

void BaseGPUDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                 OpKernelContext* context,
                                 AsyncOpKernel::DoneCallback done) {
  GPUDeviceContext* gpu_device_context = device_contexts_[0];
  if (context->op_device_context() != nullptr) {
    gpu_device_context =
        static_cast<GPUDeviceContext*>(context->op_device_context());
  }
  gpu::Stream* stream = gpu_device_context->stream();
  const auto stream_id = gpu_device_context->stream_id();

  VLOG(1) << "GpuDevice::ComputeAsync " << op_kernel->name() << " op "
          << op_kernel->type_string() << " on GPU" << gpu_id_ << " stream["
          << stream_id << "]";

  // When TraceMe profiling is off (which is the default), the
  // following TraceMe constructor is simply a conditional test of
  // false value. Measurements show that its overhead is negligible.
  port::Tracing::TraceMe activity(op_kernel->name(), op_kernel->type_string(),
                                  op_kernel->IsExpensive());
  gpu::cuda::ScopedActivateExecutorContext scoped_activation{stream->parent()};
  op_kernel->ComputeAsync(context, done);
}

Status BaseGPUDevice::MaybeCopyTensorToGPU(
    const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
    StatusCallback done) {
  if (alloc_attrs.on_host()) {
    *to = from;
    done(Status::OK());
    return Status::OK();
  } else {
    if (!DMAHelper::CanUseDMA(&from)) {
      Status err = errors::Internal("GPU copy from non-DMA ",
                                    DataTypeString(from.dtype()), " tensor");
      done(err);
      return err;
    }
    auto* copy =
        new Tensor(GetAllocator(alloc_attrs), from.dtype(), from.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy->IsInitialized()) {
      delete copy;
      Status err = errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", from.shape().DebugString(),
          " and type ", DataTypeString(from.dtype()));
      done(err);
      return err;
    }

    StatusCallback wrapped_done = std::bind(
        [to, copy](StatusCallback done_,
                   // Begin unbound arguments.
                   const Status& s) {
          *to = std::move(*copy);
          delete copy;
          done_(s);
        },
        std::move(done), std::placeholders::_1);

    port::Tracing::ScopedAnnotation annotation("MakeTensorFromProto");
    device_contexts_[0]->CopyCPUTensorToDevice(&from, this, copy,
                                               std::move(wrapped_done));
    return Status::OK();
  }
}

Status BaseGPUDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                          const AllocatorAttributes alloc_attrs,
                                          Tensor* tensor) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_alloc = GetAllocator(attr);
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  if (parsed.dtype() == DT_VARIANT) {
    if (parsed.shape().dims() != 0) {
      // TODO(b/67311047): Expand support to non-singleton variants?
      return errors::Unimplemented(
          "GPUDevice::MakeTensorFromProto: Only singleton Variants are "
          "supported. Tensor has shape: ",
          parsed.shape().DebugString());
    }
    const Variant& from = parsed.scalar<Variant>()();
    Tensor copy(cpu_allocator(), DT_VARIANT, TensorShape({}));
    Variant* copy_variant = &(copy.scalar<Variant>()());

    std::list<Notification> notifications;
    Status copy_status;
    auto copier = [this, &alloc_attrs, &notifications, &copy_status](
                      const Tensor& from, Tensor* to) {
      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToGPU(alloc_attrs, from, to,
                                  [&n, &copy_status](const Status& s) {
                                    if (copy_status.ok()) {
                                      copy_status.Update(s);
                                    }
                                    n.Notify();
                                  });
    };
    TF_RETURN_IF_ERROR(
        VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE, from,
                          copy_variant, std::move(copier)));
    for (auto& n : notifications) {
      n.WaitForNotification();
    }
    *tensor = std::move(copy);
    return copy_status;
  } else {
    Notification n;
    Status status;
    TF_RETURN_IF_ERROR(MaybeCopyTensorToGPU(alloc_attrs, parsed, tensor,
                                            [&n, &status](const Status& s) {
                                              status = s;
                                              n.Notify();
                                            }));
    n.WaitForNotification();
    return status;
  }
}

namespace {
class ConcretePerOpGpuDevice : public PerOpGpuDevice {
 public:
  ConcretePerOpGpuDevice() : device_(&stream_device_) {}

  void Reinitialize(OpKernelContext* context, const cudaStream_t* cuda_stream,
                    int gpu_id, Allocator* base_allocator, char* scratch) {
    stream_device_.Reinitialize(context, cuda_stream, gpu_id, base_allocator,
                                scratch);
  }

  const Eigen::GpuDevice& device() const override { return device_; }

 private:
  EigenCudaStreamDevice stream_device_;
  Eigen::GpuDevice device_;
};
}  // namespace

void BaseGPUDevice::ReinitializeDevice(OpKernelContext* context,
                                       PerOpGpuDevice* device, int stream_id,
                                       Allocator* allocator) {
  ConcretePerOpGpuDevice* concrete_device =
      static_cast<ConcretePerOpGpuDevice*>(device);
  DCHECK(concrete_device);
  const cudaStream_t* cuda_stream = reinterpret_cast<const cudaStream_t*>(
      streams_[stream_id]->compute->implementation()->CudaStreamMemberHack());
  concrete_device->Reinitialize(context, cuda_stream, gpu_id_, allocator,
                                scratch_[stream_id]);
}

PerOpGpuDevice* BaseGPUDevice::MakeGpuDevice() {
  return new ConcretePerOpGpuDevice();
}

void BaseGPUDevice::ReinitializeGpuDevice(OpKernelContext* context,
                                          PerOpGpuDevice* device,
                                          DeviceContext* dc,
                                          Allocator* allocator) {
  if (dc) {
    const GPUDeviceContext* gpu_dc = static_cast<GPUDeviceContext*>(dc);
    const int stream_id = gpu_dc->stream_id();
    VLOG(1) << "  eigen_gpu_device(" << dc << ") => stream[" << stream_id
            << "]";
    CHECK_LT(stream_id, streams_.size());
    ReinitializeDevice(context, device, stream_id, allocator);
  } else {
    ReinitializeDevice(context, device, 0, allocator);
  }
}

Status BaseGPUDeviceFactory::CreateDevices(const SessionOptions& options,
                                           const string& name_prefix,
                                           std::vector<Device*>* devices) {
  size_t n = INT_MAX;
  auto iter = options.config.device_count().find("GPU");
  if (iter != options.config.device_count().end()) {
    n = iter->second;
  }
  std::vector<int> valid_gpu_ids;
  TF_RETURN_IF_ERROR(GetValidDeviceIds(
      options.config.gpu_options().visible_device_list(), &valid_gpu_ids));
  if (static_cast<size_t>(n) > valid_gpu_ids.size()) {
    n = valid_gpu_ids.size();
  }
  if (!valid_gpu_ids.empty()) {
    // Save the original device.
    int original_device = 0;
    cudaError_t err = cudaGetDevice(&original_device);
    if (err != cudaSuccess) {
      return errors::Internal("cudaGetDevice() failed. Status: ",
                              cudaGetErrorString(err));
    }
    // Force to implicitly initialize CUDA runtime on each valid GPU before
    // CreateGPUDevice().
    for (int gpu_id : valid_gpu_ids) {
      err = cudaSetDevice(gpu_id);
      if (err != cudaSuccess) {
        return errors::Internal("cudaSetDevice() on GPU:", gpu_id,
                                " failed. Status: ", cudaGetErrorString(err));
      }
      err = cudaFree(nullptr);
      if (err != cudaSuccess) {
        return errors::Internal(
            "CUDA runtime implicit initialization on GPU:", gpu_id,
            " failed. Status: ", cudaGetErrorString(err));
      }
    }
    // Reset to the original device.
    err = cudaSetDevice(original_device);
    if (err != cudaSuccess) {
      return errors::Internal("cudaSetDevice() on GPU:", original_device,
                              " failed. Status: ", cudaGetErrorString(err));
    }
  }
  for (int i = 0; i < n; i++) {
    BaseGPUDevice* gpu_device;
    TF_RETURN_IF_ERROR(CreateGPUDevice(
        options, strings::StrCat(name_prefix, "/device:GPU:", i),
        valid_gpu_ids[i], &gpu_device));
    TF_RETURN_IF_ERROR(gpu_device->Init(options));
    devices->push_back(gpu_device);
  }

  return Status::OK();
}

namespace {
int64 MinSystemMemory(int64 available_memory) {
  // We use the following heuristic for now:
  //
  // If the available_memory is < 2GiB, we allocate 225MiB to system memory.
  // Otherwise, allocate max(300MiB, 0.05 * available_memory) to system memory.
  //
  // In the future we could be more sophisticated by using a table of devices.
  int64 min_system_memory;
  if (available_memory < (1LL << 31)) {
    // 225MiB
    min_system_memory = 225 * 1024 * 1024;
  } else {
    // max(300 MiB, 0.05 * available_memory)
    min_system_memory =
        std::max(314572800LL, static_cast<int64>(available_memory * 0.05));
  }
#if defined(__GNUC__) && defined(__OPTIMIZE__)
// Do nothing
#elif !defined(__GNUC__) && defined(NDEBUG)
// Do nothing
#else
  // Double the amount of available GPU memory in non-opt builds (debug
  // builds in windows); because in non-opt builds more system memory
  // is necessary.
  min_system_memory *= 2;
#endif
  return min_system_memory;
}

}  // namespace

static string GetShortDeviceDescription(int device_id,
                                        const gpu::DeviceDescription& desc) {
  int cc_major;
  int cc_minor;
  if (!desc.cuda_compute_capability(&cc_major, &cc_minor)) {
    cc_major = 0;
    cc_minor = 0;
  }
  // LINT.IfChange
  return strings::StrCat("device: ", device_id, ", name: ", desc.name(),
                         ", pci bus id: ", desc.pci_bus_id(),
                         ", compute capability: ", cc_major, ".", cc_minor);
  // LINT.ThenChange(//tensorflow/python/platform/test.py)
}

Status BaseGPUDeviceFactory::CreateGPUDevice(const SessionOptions& options,
                                             const string& name, int gpu_id,
                                             BaseGPUDevice** out_device) {
  CHECK_GE(gpu_id, 0);

  // Look up the device, to see its attributes.
  gpu::Platform* gpu_platform = GPUMachineManager();
  CHECK_LT(gpu_id, gpu_platform->VisibleDeviceCount());
  gpu::StreamExecutor* se =
      gpu_platform->ExecutorForDevice(gpu_id).ValueOrDie();
  const gpu::DeviceDescription& desc = se->GetDeviceDescription();
  int numa_node = desc.numa_node();
  if (numa_node < 0) {
    // For some reason the StreamExecutor couldn't get the NUMA
    // affinity of the GPU.  If this is not a multi-socket mobo with
    // GPUs local to different buses, it doesn't matter.  If it is, we
    // may run into trouble later with data transfer operations.  The
    // trouble may manifest as slower than expected performance, or
    // outright failures.
    LOG(INFO) << "Could not identify NUMA node of " << name
              << ", defaulting to 0.  Your kernel may not have been built "
              << "with NUMA support.";
    numa_node = 0;
  }

  int64 total_memory, available_memory;
  if (!se->DeviceMemoryUsage(&available_memory, &total_memory)) {
    return errors::Unknown(
        strings::StrCat("Failed to query available memory for GPU ", gpu_id));
  }

  int64 allocated_memory;
  double config_memory_fraction =
      options.config.gpu_options().per_process_gpu_memory_fraction();
  if (config_memory_fraction == 0) {
    allocated_memory = available_memory;
    const int64 min_system_memory = MinSystemMemory(available_memory);
    if (min_system_memory < allocated_memory) {
      allocated_memory -= min_system_memory;
    }
  } else {
    allocated_memory = total_memory * config_memory_fraction;
  }

  Bytes allocated_bytes = static_cast<Bytes>(allocated_memory);

  // Get GPU bus_id from its reported NUMA affinity.  Because GPUs are
  // virtualized in some environments, we can't just use the GPU id.
  // NUMA locales are indexed from 0, buses are indexed from 1.
  DeviceLocality dev_locality;
  dev_locality.set_bus_id(numa_node + 1);
  VLOG(1) << "GPUDevice id " << gpu_id << " on bus " << dev_locality.bus_id()
          << " numa: " << numa_node << " pci: " << desc.pci_bus_id();

  ProcessState* process_state = ProcessState::singleton();
  *out_device = CreateGPUDevice(
      options, name, allocated_bytes, dev_locality, gpu_id,
      GetShortDeviceDescription(gpu_id, desc),
      process_state->GetGPUAllocator(options.config.gpu_options(), gpu_id,
                                     allocated_memory),
      process_state->GetCPUAllocator(numa_node));

  return Status::OK();
}

static int GetDefaultMinGPUMultiprocessorCount(
    gpu::Platform* gpu_manager, const std::vector<int>& visible_gpu_order) {
  static const int kDefaultMinGPUMultiprocessorCount = 8;

  // Find the highest multi-processor count across all visible GPUs.
  int max_count = -1;
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    auto exec_status = gpu_manager->ExecutorForDevice(visible_gpu_order[i]);
    if (!exec_status.ok()) {
      continue;
    }

    gpu::StreamExecutor* se = exec_status.ValueOrDie();
    const gpu::DeviceDescription& desc = se->GetDeviceDescription();
    max_count = std::max(max_count, desc.core_count());
  }

  if (max_count < 0 || kDefaultMinGPUMultiprocessorCount < max_count) {
    return kDefaultMinGPUMultiprocessorCount;
  } else {
    return max_count;
  }
}

static int GetMinGPUMultiprocessorCount(
    gpu::Platform* gpu_manager, const std::vector<int>& visible_gpu_order) {
  const char* tf_min_gpu_core_count = getenv("TF_MIN_GPU_MULTIPROCESSOR_COUNT");

  if (tf_min_gpu_core_count == nullptr ||
      strcmp(tf_min_gpu_core_count, "") == 0) {
    return GetDefaultMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);
  }

  int min_gpu_core_count = -1;
  if (strings::safe_strto32(tf_min_gpu_core_count, &min_gpu_core_count)) {
    if (min_gpu_core_count >= 0) {
      return min_gpu_core_count;
    }
  }

  int count =
      GetDefaultMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);
  LOG(ERROR) << "Invalid minimum GPU multiprocessor count: ["
             << tf_min_gpu_core_count << "]. "
             << "Using the default value: " << count;
  return count;
}

namespace {

struct CudaVersion {
  // Initialize from version_name in the form of "3.5"
  explicit CudaVersion(const std::string& version_name) {
    size_t dot_pos = version_name.find('.');
    CHECK(dot_pos != string::npos)
        << "Illegal version name: [" << version_name << "]";
    string major_str = version_name.substr(0, dot_pos);
    CHECK(strings::safe_strto32(major_str, &major_part))
        << "Illegal version name: [" << version_name << "]";
    string minor_str = version_name.substr(dot_pos + 1);
    CHECK(strings::safe_strto32(minor_str, &minor_part))
        << "Illegal version name: [" << version_name << "]";
  }
  CudaVersion() {}
  bool operator<(const CudaVersion& other) const {
    if (this->major_part != other.major_part) {
      return this->major_part < other.major_part;
    }
    return this->minor_part < other.minor_part;
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const CudaVersion& version) {
    os << version.major_part << "." << version.minor_part;
    return os;
  }
  int major_part = -1;
  int minor_part = -1;
};

std::vector<CudaVersion> supported_cuda_compute_capabilities = {
    TF_CUDA_CAPABILITIES,};

std::vector<CudaVersion> GetSupportedCudaComputeCapabilities() {
  auto cuda_caps = supported_cuda_compute_capabilities;
#ifdef TF_EXTRA_CUDA_CAPABILITIES
// TF_EXTRA_CUDA_CAPABILITIES should be defined a sequence separated by commas,
// for example:
//   TF_EXTRA_CUDA_CAPABILITIES=3.0,4.0,5.0
// Use two-level macro expansion for stringification.
#define TF_XSTRING(...) #__VA_ARGS__
#define TF_STRING(s) TF_XSTRING(s)
  string extra_cuda_caps = TF_STRING(TF_EXTRA_CUDA_CAPABILITIES);
#undef TF_STRING
#undef TF_XSTRING
  auto extra_capabilities = str_util::Split(extra_cuda_caps, ',');
  for (const auto& capability : extra_capabilities) {
    cuda_caps.push_back(CudaVersion(capability));
  }
#endif
  return cuda_caps;
}

std::unique_ptr<std::map<std::pair<int, int>, bool>> GetPeerAccessMap(
    gpu::Platform* platform, const std::vector<int>& visible_gpu_order) {
  std::unique_ptr<std::map<std::pair<int, int>, bool>> map(
      new std::map<std::pair<int, int>, bool>);
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    const int i_gpu_id = visible_gpu_order[i];
    for (int j = 0; j < visible_gpu_order.size(); ++j) {
      const int j_gpu_id = visible_gpu_order[j];
      gpu::StreamExecutor* from =
          platform->ExecutorForDevice(i_gpu_id).ValueOrDie();
      gpu::StreamExecutor* to =
          platform->ExecutorForDevice(j_gpu_id).ValueOrDie();
      (*map)[{i, j}] = from->CanEnablePeerAccessTo(to);
    }
  }

  return map;
}

Status EnablePeerAccess(gpu::Platform* platform,
                        const std::vector<int>& visible_gpu_order) {
  int possible_peer_count = 0;
  int enabled_peer_count = 0;
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    const int i_gpu_id = visible_gpu_order[i];
    for (int j = 0; j < visible_gpu_order.size(); ++j) {
      const int j_gpu_id = visible_gpu_order[j];
      // We have already validated that ExecutorForDevice() calls
      // return OK.
      gpu::StreamExecutor* from =
          platform->ExecutorForDevice(i_gpu_id).ValueOrDie();
      gpu::StreamExecutor* to =
          platform->ExecutorForDevice(j_gpu_id).ValueOrDie();

      if (from->CanEnablePeerAccessTo(to)) {
        ++possible_peer_count;
        auto status = from->EnablePeerAccessTo(to);
        if (!status.ok()) {
          LOG(WARNING)
              << "Unable to enable peer access between device ordinals "
              << i_gpu_id << " and " << j_gpu_id;
        } else {
          ++enabled_peer_count;
        }
      }
    }
  }

  // Return an error in the extreme failure case where the driver
  // reported that peering was possible but not a single peering was
  // successful.  This is to catch possible system misconfigurations
  // or more fundamental issues.
  if (possible_peer_count > 0 && enabled_peer_count == 0) {
    return errors::Internal(possible_peer_count,
                            " potential peer access pairs were reported by the "
                            "driver, but no peering could be enabled.");
  }
  return Status::OK();
}

}  // namespace

Status BaseGPUDeviceFactory::GetValidDeviceIds(
    const string& visible_device_list, std::vector<int>* ids) {
  TF_RETURN_IF_ERROR(ValidateGPUMachineManager());

  gpu::Platform* gpu_manager = GPUMachineManager();
  if (gpu_manager == nullptr) {
    return Status::OK();
  }

  // If there are no GPUs visible, do nothing.
  if (gpu_manager->VisibleDeviceCount() <= 0) {
    return Status::OK();
  }

  // If the user wants to remap the visible to virtual GPU mapping,
  // check for that here.
  std::vector<int> visible_gpu_order;
  if (visible_device_list.empty()) {
    visible_gpu_order.resize(gpu_manager->VisibleDeviceCount());
    // By default, visible to virtual mapping is unchanged.
    int deviceNo = 0;
    std::generate(visible_gpu_order.begin(), visible_gpu_order.end(),
                  [&deviceNo] { return deviceNo++; });
  } else {
    std::vector<string> order_str = str_util::Split(visible_device_list, ',');
    for (int i = 0; i < order_str.size(); ++i) {
      const string& gpu_id_str = order_str[i];
      int32 gpu_id;
      if (!strings::safe_strto32(gpu_id_str, &gpu_id)) {
        return errors::InvalidArgument(
            "Could not parse entry in 'visible_device_list': '", gpu_id_str,
            "'.  visible_device_list = ", visible_device_list);
      }

      if (gpu_id < 0 || gpu_id >= gpu_manager->VisibleDeviceCount()) {
        return errors::InvalidArgument(
            "'visible_device_list' listed an invalid GPU id '", gpu_id,
            "' but visible device count is ",
            gpu_manager->VisibleDeviceCount());
      }

      visible_gpu_order.push_back(gpu_id);
    }
  }

  // Validate no repeats.
  std::set<int> visible_device_set(visible_gpu_order.begin(),
                                   visible_gpu_order.end());
  if (visible_device_set.size() != visible_gpu_order.size()) {
    return errors::InvalidArgument(
        "visible_device_list contained "
        "a duplicate entry: ",
        visible_device_list);
  }

  bool new_gpu_found = false;
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    int gpu_id = visible_gpu_order[i];

    // Only perform this once per visible gpu id.
    if (visible_gpu_initialized_[gpu_id]) {
      continue;
    }

    visible_gpu_initialized_[gpu_id] = true;
    new_gpu_found = true;

    auto executor = gpu_manager->ExecutorForDevice(gpu_id);
    if (!executor.ok()) {
      return StreamExecutorUtil::ConvertStatus(executor.status());
    }

    auto stream_exec = executor.ValueOrDie();
    int64 free_bytes;
    int64 total_bytes;
    if (!stream_exec->DeviceMemoryUsage(&free_bytes, &total_bytes)) {
      // Logs internally on failure.
      free_bytes = 0;
      total_bytes = 0;
    }
    const auto& description = stream_exec->GetDeviceDescription();
    int cc_major;
    int cc_minor;
    if (!description.cuda_compute_capability(&cc_major, &cc_minor)) {
      // Logs internally on failure.
      cc_major = 0;
      cc_minor = 0;
    }
    LOG(INFO) << "Found device " << i << " with properties: "
              << "\nname: " << description.name() << " major: " << cc_major
              << " minor: " << cc_minor
              << " memoryClockRate(GHz): " << description.clock_rate_ghz()
              << "\npciBusID: " << description.pci_bus_id() << "\ntotalMemory: "
              << strings::HumanReadableNumBytes(total_bytes)
              << " freeMemory: " << strings::HumanReadableNumBytes(free_bytes);
  }
  // Checking peering and shows matrix if more than one gpu found.
  if (new_gpu_found && visible_gpu_order.size() > 1) {
    // Enable peer access
    TF_RETURN_IF_ERROR(EnablePeerAccess(gpu_manager, visible_gpu_order));

    // Print out a matrix showing which devices can DMA to one
    // another.
    LOG(INFO) << "Device peer to peer matrix";
    auto access_map = GetPeerAccessMap(gpu_manager, visible_gpu_order);
    string line_buf = "DMA: ";
    for (int i = 0; i < visible_gpu_order.size(); ++i) {
      strings::StrAppend(&line_buf, visible_gpu_order[i], " ");
    }
    LOG(INFO) << line_buf;
    for (int i = 0; i < visible_gpu_order.size(); ++i) {
      line_buf = strings::StrCat(visible_gpu_order[i], ":   ");
      for (int j = 0; j < visible_gpu_order.size(); ++j) {
        if ((*access_map)[{i, j}]) {
          line_buf.append("Y ");
        } else {
          line_buf.append("N ");
        }
      }
      LOG(INFO) << line_buf;
    }
  }

  auto cuda_supported_capabilities = GetSupportedCudaComputeCapabilities();
  if (cuda_supported_capabilities.empty()) {
    return errors::FailedPrecondition(
        "No supported cuda capabilities in binary.");
  }
  CudaVersion min_supported_capability = *std::min_element(
      cuda_supported_capabilities.begin(), cuda_supported_capabilities.end());

  int min_gpu_core_count =
      GetMinGPUMultiprocessorCount(gpu_manager, visible_gpu_order);

  // Filter out devices that don't have the right capability or power.
  for (int i = 0; i < visible_gpu_order.size(); ++i) {
    const int32 visible_gpu_id = visible_gpu_order[i];
    auto exec_status = gpu_manager->ExecutorForDevice(visible_gpu_id);
    if (!exec_status.ok()) {
      continue;
    }
    gpu::StreamExecutor* se = exec_status.ValueOrDie();
    const gpu::DeviceDescription& desc = se->GetDeviceDescription();
    CudaVersion device_capability;
    if (!desc.cuda_compute_capability(&device_capability.major_part,
                                      &device_capability.minor_part)) {
      continue;
    }
    // Only GPUs with no less than the minimum supported compute capability is
    // accepted.
    if (device_capability < min_supported_capability) {
      LOG(INFO) << "Ignoring visible gpu device "
                << "(" << GetShortDeviceDescription(visible_gpu_id, desc)
                << ") "
                << "with Cuda compute capability " << device_capability
                << ". The minimum required Cuda capability is "
                << min_supported_capability << ".";
      continue;
    }

    // Filter out slow GPUs. By default, GPUs with a lower multiprocessor
    // count than the fastest GPU are filtered out, unless they have 8 or more
    // multiprocessors. If the TF_MIN_GPU_MULTIPROCESSOR_COUNT environment
    // variable is set, its value will be used to filter out GPUs.
    if (desc.core_count() < min_gpu_core_count) {
      LOG(INFO) << "Ignoring gpu device "
                << "(" << GetShortDeviceDescription(visible_gpu_id, desc)
                << ") "
                << "with Cuda multiprocessor count: " << desc.core_count()
                << ". The minimum required count is " << min_gpu_core_count
                << ". You can adjust this requirement with the env var "
                   "TF_MIN_GPU_MULTIPROCESSOR_COUNT.";
      continue;
    }

    size_t new_id = ids->size();
    ids->push_back(visible_gpu_id);

    LOG(INFO) << "Creating TensorFlow device (/device:GPU:" << new_id << ") -> "
              << "(" << GetShortDeviceDescription(visible_gpu_id, desc) << ")";
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
