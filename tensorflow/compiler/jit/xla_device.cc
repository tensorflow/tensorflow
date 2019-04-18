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

#include "tensorflow/compiler/jit/xla_device.h"

#include <stdlib.h>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

// Caches a XlaDeviceAllocator per <backend, device ordinal> pair. A
// XlaDeviceAllocator is created on demand and is associated with a
// XlaDevice. It outlives the device itself (for instance, the buffer
// backing a tensor holds a pointer to the allocator for book-keeping,
// and this buffer can outlast the device).
class XlaDeviceAllocatorState {
 public:
  // Creates or returns a cached XlaDeviceAllocator for a given
  // backend and device_ordinal.
  static XlaDeviceAllocator* GetOrCreateXlaDeviceAllocator(
      const xla::Backend* backend, int device_ordinal);

 private:
  // Returns the singleton instance of XlaDeviceAllocatorState.
  static XlaDeviceAllocatorState& Singleton();
  XlaDeviceAllocatorState();
  ~XlaDeviceAllocatorState();

  mutex allocator_mutex_;  // Guards the singleton allocator state.
  std::unordered_map<std::pair<const xla::Backend*, int>,
                     std::unique_ptr<XlaDeviceAllocator>,
                     hash<std::pair<const xla::Backend*, int>>>
      allocators_ GUARDED_BY(allocator_mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaDeviceAllocatorState);
};

/* static */ XlaDeviceAllocatorState& XlaDeviceAllocatorState::Singleton() {
  static auto a = new XlaDeviceAllocatorState;
  return *a;
}

XlaDeviceAllocatorState::XlaDeviceAllocatorState() = default;
XlaDeviceAllocatorState::~XlaDeviceAllocatorState() = default;

XlaDeviceAllocator* XlaDeviceAllocatorState::GetOrCreateXlaDeviceAllocator(
    const xla::Backend* backend, int device_ordinal) {
  XlaDeviceAllocatorState& state = Singleton();
  mutex_lock lock(state.allocator_mutex_);

  auto it = state.allocators_.find({backend, device_ordinal});
  if (it != state.allocators_.end()) {
    return it->second.get();
  }

  std::unique_ptr<XlaDeviceAllocator> alloc =
      absl::make_unique<XlaDeviceAllocator>(
          backend->stream_executors()[device_ordinal]);
  XlaDeviceAllocator* alloc_ptr = alloc.get();
  state.allocators_[{backend, device_ordinal}] = std::move(alloc);
  return alloc_ptr;
}

namespace {

// Default PaddedShapeFn implementation that simply returns the unpadded
// on-device shape. This is accurate for CPU and GPU devices that neither
// transpose nor pad tensors.
Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape) {
  const tensorflow::XlaTensor* xla_tensor =
      tensorflow::XlaTensor::FromTensor(&tensor);
  if (xla_tensor == nullptr) {
    return TensorShapeToXLAShape(tensor.dtype(), tensor.shape(), shape);
  }

  const xla::ShapedBuffer& shaped_buffer = xla_tensor->shaped_buffer();
  *shape = shaped_buffer.on_device_shape();
  return Status::OK();
}

static DeviceAttributes BuildXlaDeviceAttributes(const string& name_prefix,
                                                 const string& device_name,
                                                 int device_ordinal) {
  return Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", device_name, ":", device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      absl::StrCat("device: ", device_name, " device"));
}

}  // namespace

XlaDevice::Metadata::Metadata(
    int device_ordinal, se::Platform* platform, const DeviceType& device_type,
    XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    PaddedShapeFn padded_shape_fn, bool use_multiple_streams)
    : device_ordinal_(device_ordinal),
      device_type_(device_type),
      platform_(platform),
      shape_representation_fn_(std::move(shape_representation_fn)),
      padded_shape_fn_(std::move(padded_shape_fn)),
      use_multiple_streams_(use_multiple_streams) {}

int XlaDevice::Metadata::device_ordinal() const { return device_ordinal_; }

se::Platform* XlaDevice::Metadata::platform() const { return platform_; }

xla::LocalClient* XlaDevice::Metadata::client() const {
  auto client = xla::ClientLibrary::GetOrCreateLocalClient(platform_);
  return client.ValueOrDie();
}

const DeviceType& XlaDevice::Metadata::jit_device_type() const {
  return device_type_;
}

/*static*/ Status XlaDevice::GetMetadataFromDevice(
    DeviceBase* device, const XlaDevice::Metadata** metadata) {
  *metadata = nullptr;
  XlaDevice* xla_device = dynamic_cast<XlaDevice*>(device->UnderlyingDevice());
  if (xla_device == nullptr) {
    return errors::Internal(
        "Cannot get XLA metadata from non-XLA device \"", device->name(),
        "\". GetMetadata must only be called on an XLA device. Either an "
        "internal bug has been triggered, or an XLA-specific op has been "
        "placed on the wrong device.");
  }
  *metadata = &(xla_device->xla_metadata_);
  return Status::OK();
}

/* static */ Status XlaDevice::GetMetadata(OpKernelContext* ctx,
                                           const Metadata** metadata) {
  return GetMetadataFromDevice(ctx->device(), metadata);
}

/* static */ Status XlaDevice::GetMetadata(OpKernelConstruction* ctx,
                                           const Metadata** metadata) {
  return GetMetadataFromDevice(ctx->device(), metadata);
}

XlaDevice::XlaDevice(const SessionOptions& session_options,
                     const Options& options)
    : LocalDevice(session_options,
                  BuildXlaDeviceAttributes(options.device_name_prefix,
                                           options.device_name,
                                           options.device_ordinal)),
      xla_metadata_(options.device_ordinal, options.platform,
                    DeviceType(options.compilation_device_name),
                    options.shape_representation_fn,
                    options.padded_shape_fn ? options.padded_shape_fn
                                            : DefaultPaddedShapeFn,
                    options.use_multiple_streams),
      device_ordinal_(options.device_ordinal),
      jit_device_name_(options.compilation_device_name),
      platform_(options.platform),
      use_multiple_streams_(options.use_multiple_streams),
      shape_representation_fn_(options.shape_representation_fn),
      allowed_devices_(options.allowed_devices) {
  VLOG(1) << "Created XLA device " << options.compilation_device_name << " "
          << this;
  thread_pool_.reset(new thread::ThreadPool(session_options.env, "xla_device",
                                            /*num_threads=*/1));

  // We have multiple device to device streams to allow for some concurrency
  // between transfers. The particular value of '4' is chosen fairly
  // arbitrarily. It may be necessary to make this tunable via
  // XlaDevice::Options.
  static constexpr int kNumDeviceToDeviceStreams = 4;
  device_to_device_streams_.resize(kNumDeviceToDeviceStreams);
}

XlaDevice::~XlaDevice() {
  VLOG(1) << "Destroying XLA device " << jit_device_name_ << " " << this;
  mutex_lock lock(mu_);
  if (device_context_) {
    device_context_->Unref();
  }
}

xla::LocalClient* XlaDevice::client() const {
  // We lazily create the client because the platform commits to the
  // details of the host hardware when the client is created, so we
  // don't want to do it until we get a chance to hook the platform up
  // to a simulator.

  // TODO(b/78468222): This can fail, at least when the backend is GPU and
  // there is no GPU on the host.
  return xla::ClientLibrary::GetOrCreateLocalClient(platform_, allowed_devices_)
      .ValueOrDie();
}

Allocator* XlaDevice::GetAllocator(AllocatorAttributes attr) {
  mutex_lock lock(mu_);
  return GetAllocatorLocked(attr);
}

Allocator* XlaDevice::GetAllocatorLocked(AllocatorAttributes attr) {
  if (attr.on_host()) {
    return cpu_allocator();
  }

  if (xla_allocator_ == nullptr) {
    xla::Backend* backend = client()->mutable_backend();
    xla_allocator_ = XlaDeviceAllocatorState::GetOrCreateXlaDeviceAllocator(
        backend, device_ordinal_);
  }
  return xla_allocator_;
}

Status XlaDevice::EnsureDeviceContextOk() {
  mutex_lock lock(mu_);
  return GetDeviceContextLocked().status();
}

Status XlaDevice::EnsureStreamOkLocked(xla::Backend* backend,
                                       const string& name,
                                       std::shared_ptr<se::Stream>* stream,
                                       bool* stream_was_changed) {
  if (!(*stream) || !(*stream)->ok()) {
    xla::StreamPool::Ptr ptr;
    TF_ASSIGN_OR_RETURN(ptr, backend->BorrowStream(device_ordinal_));
    *stream = std::shared_ptr<se::Stream>(std::move(ptr));
    VLOG(1) << "XlaDevice " << this << " new " << name << " "
            << (*stream)->DebugStreamPointers();
    *stream_was_changed = true;
  }
  return Status::OK();
}

xla::StatusOr<XlaDeviceContext*> XlaDevice::GetDeviceContextLocked() {
  xla::Backend* backend = client()->mutable_backend();

  // Ensure all our streams are valid, borrowing new streams if necessary.
  bool need_new_device_context = !device_context_;
  TF_RETURN_IF_ERROR(EnsureStreamOkLocked(backend, "stream", &stream_,
                                          &need_new_device_context));

  std::shared_ptr<se::Stream> host_to_device_stream;
  std::shared_ptr<se::Stream> device_to_host_stream;
  std::vector<std::shared_ptr<se::Stream>> device_to_device_streams;
  if (use_multiple_streams_) {
    TF_RETURN_IF_ERROR(EnsureStreamOkLocked(backend, "host_to_device_stream",
                                            &host_to_device_stream_,
                                            &need_new_device_context));
    for (std::shared_ptr<se::Stream>& stream : device_to_device_streams_) {
      TF_RETURN_IF_ERROR(
          EnsureStreamOkLocked(backend, "device_to_device_stream", &stream,
                               &need_new_device_context));
    }
    host_to_device_stream = host_to_device_stream_;
    device_to_device_streams = device_to_device_streams_;
    // The data transfer requests from device to host could arrive out of order,
    // so a single stream would cause deadlock. For this case,
    // xla_device_context would borrow a stream for each transfer request.
    device_to_host_stream = nullptr;
  } else {
    host_to_device_stream = stream_;
    device_to_host_stream = stream_;
    device_to_device_streams = {stream_};
  }

  if (!need_new_device_context) {
    return device_context_;
  }

  // At this point we know we need a new device context.
  // Call GetAllocator for the side-effect of ensuring the allocator is created.
  GetAllocatorLocked({});
  if (device_context_) {
    device_context_->Unref();
  }
  // The XlaDeviceContext keeps a reference count to the streams, and the
  // XlaDeviceContext remains live for the duration of a Executor run. This
  // ensures that the streams remain live for the duration of a run, even if
  // an error is encountered and the streams are replaced with new ones.
  device_context_ = new XlaDeviceContext(
      stream_, std::move(host_to_device_stream),
      std::move(device_to_host_stream), std::move(device_to_device_streams),
      client(), shape_representation_fn_, thread_pool_.get());
  VLOG(1) << "XlaDevice " << this << " new XlaDeviceContext "
          << device_context_;

  // Create and set a new GpuDeviceInfo, if necessary.
  //
  // TODO(b/78232898): This isn't thread-safe; there is a race between the call
  // to set_tensorflow_gpu_device_info() with ops that call the getter
  // tensorflow_gpu_device_info(). This isn't trivially fixed by adding locking
  // to those methods; see the bug for details. Our only saving grace at the
  // moment is that this race doesn't seem to occur in practice.
  if (use_gpu_device_info_) {
    auto gpu_device_info = absl::make_unique<GpuDeviceInfo>();
    gpu_device_info->stream = stream_.get();
    gpu_device_info->default_context = device_context_;
    set_tensorflow_gpu_device_info(gpu_device_info.get());
    gpu_device_info_ = std::move(gpu_device_info);
    VLOG(1) << "XlaDevice " << this << " new GpuDeviceInfo "
            << gpu_device_info_.get();
  }

  return device_context_;
}

Status XlaDevice::UseGpuDeviceInfo() {
  mutex_lock lock(mu_);
  use_gpu_device_info_ = true;
  return GetDeviceContextLocked().status();
}

Status XlaDevice::FillContextMap(const Graph* graph,
                                 DeviceContextMap* device_context_map) {
  VLOG(1) << "XlaDevice::FillContextMap";
  mutex_lock lock(mu_);
  TF_ASSIGN_OR_RETURN(XlaDeviceContext * device_context,
                      GetDeviceContextLocked());

  device_context_map->resize(graph->num_node_ids());
  for (Node* n : graph->nodes()) {
    VLOG(2) << n->id() << " : " << n->type_string() << " : " << n->name();
    device_context->Ref();
    (*device_context_map)[n->id()] = device_context;
  }
  return Status::OK();
}

void XlaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  VLOG(2) << "XlaDevice::Compute " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->Compute(context);
}

void XlaDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                             AsyncOpKernel::DoneCallback done) {
  VLOG(2) << "XlaDevice::ComputeAsync " << op_kernel->name() << ":"
          << op_kernel->type_string();
  tracing::ScopedActivity activity(op_kernel->name(), op_kernel->type_string(),
                                   op_kernel->IsExpensive());
  op_kernel->ComputeAsync(context, done);
}

Status XlaDevice::Sync() {
  VLOG(1) << "XlaDevice::Sync";
  tracing::ScopedActivity activity("XlaDevice::Sync", /*is_expensive=*/true);
  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) return Status::OK();

  Status status = stream->BlockHostUntilDone();
  TF_RETURN_IF_ERROR(status);
  if (!stream->ok()) {
    return errors::Internal("XlaDevice::Sync() failed.");
  }
  VLOG(1) << "XlaDevice::Sync completed";
  return Status::OK();
}

// TODO(b/112409994): This is no longer necessary. Consolidate it with the
// synchronous version.
void XlaDevice::Sync(const DoneCallback& done) {
  VLOG(1) << "XlaDevice::Sync (asynchronous)";
  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) {
    done(Status::OK());
    return;
  }

  // The call to ThenEnqueueOnBackgroundThread below enqueues a host callback at
  // the end of the stream, after everything that has already been enqueued
  // there at this moment. When the host callback is called, everything before
  // it must have already finished, and the host callback will then place the
  // task below onto a background thread. (See the implementation of
  // ThenEnqueueOnBackgroundThread for details.) Therefore, when the done
  // callback is finally called from that background thread, we know for sure
  // that everything enqueued onto the stream (i.e., the device) at this very
  // moment--when ThenEnqueueOnBackgroundThread is called--will have finished.
  // This achieves a device-wide sync.
  stream->ThenEnqueueOnBackgroundThread(
      [stream, done](se::StreamExecutor*) {
        tracing::ScopedActivity activity("XlaDevice::Sync::Callback",
                                         /*is_expensive=*/true);
        done(stream->ok() ? Status::OK()
                          : errors::Internal("XlaDevice::Sync() failed."));
      });
}

Status XlaDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                      const AllocatorAttributes alloc_attrs,
                                      Tensor* tensor) {
  VLOG(1) << "XlaDevice::MakeTensorFromProto";

  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    mutex_lock lock(mu_);
    TF_ASSIGN_OR_RETURN(XlaDeviceContext * device_context,
                        GetDeviceContextLocked());
    Allocator* allocator = GetAllocatorLocked(alloc_attrs);
    Tensor copy(allocator, parsed.dtype(), parsed.shape());
    Notification n;
    device_context->CopyCPUTensorToDevice(&parsed, this, &copy,
                                          [&n, &status](const Status& s) {
                                            status = s;
                                            n.Notify();
                                          });
    n.WaitForNotification();
    *tensor = copy;
  }
  VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
  return status;
}

void XlaDevice::SetAllowsSyncOnCompletion(bool sync_on_completion) {
  mutex_lock lock(mu_);
  sync_on_completion_ = sync_on_completion;
}

bool XlaDevice::AllowsSyncOnCompletion() const {
  mutex_lock lock(mu_);
  return sync_on_completion_;
}

void XlaDevice::SetHandleDeviceErrorCallback(std::function<Status()> callback) {
  mutex_lock lock(mu_);
  device_error_callback_ = callback;
}

Status XlaDevice::HandleDeviceError() {
  std::function<Status()> local_device_error_callback;
  {
    mutex_lock lock(mu_);
    local_device_error_callback = device_error_callback_;
  }
  if (local_device_error_callback != nullptr) {
    return local_device_error_callback();
  }
  return Status::OK();
}

Status XlaDevice::RefreshStatus() {
  std::shared_ptr<se::Stream> stream;
  {
    mutex_lock lock(mu_);
    stream = stream_;
  }
  if (!stream) {
    return Status::OK();
  }
  Status status = stream->RefreshStatus();
  if (!status.ok()) {
    // Ignore errors from HandleDeviceError, since by definition the status is
    // already non-ok, so there's nothing extra to report if HandleDeviceError
    // itself returns an error.
    HandleDeviceError().IgnoreError();
  }
  return status;
}

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device) {
  // Any op assigned to the device that isn't rewritten by the graph rewriter
  // gets executed by an XlaCompileOnDemandOp, which compiles it and executes
  // it just-in-time.
  OpKernel* (*factory)(OpKernelConstruction*) =
      [](OpKernelConstruction* context) -> OpKernel* {
    return new XlaCompileOnDemandOp(context);
  };
  XlaOpRegistry::RegisterCompilationKernels();
  XlaDeviceOpRegistrations* registrations = new XlaDeviceOpRegistrations;
  for (const KernelDef* jit_def : XlaOpRegistry::DeviceKernels(
           jit_device,
           /*include_compilation_only_kernels=*/false)) {
    KernelDef* def = new KernelDef(*jit_def);
    def->set_device_type(device);
    registrations->op_kernel_registrars.emplace_back(
        new kernel_factory::OpKernelRegistrar(def, "XlaCompileOnDemandOp",
                                              factory));
  }
  return registrations;
}

}  // namespace tensorflow
