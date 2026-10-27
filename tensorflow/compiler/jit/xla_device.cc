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

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/pjrt_device_context.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_tensor.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tfrt/common/async_value_tensor.h"

namespace tensorflow {

// Default PaddedShapeFn implementation that simply returns the unpadded
// on-device shape. This is accurate for CPU and GPU devices that neither
// transpose nor pad tensors.
absl::Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape) {
  const tensorflow::XlaTensor* xla_tensor =
      tensorflow::XlaTensor::FromTensor(&tensor);
  if (xla_tensor == nullptr) {
    return TensorShapeToXLAShape(tensor.dtype(), tensor.shape(), shape);
  }

  const xla::ShapedBuffer& shaped_buffer = xla_tensor->shaped_buffer();
  *shape = shaped_buffer.on_device_shape();
  return absl::OkStatus();
}


namespace {

static DeviceAttributes BuildXlaDeviceAttributes(const std::string& name_prefix,
                                                 const std::string& device_name,
                                                 int device_ordinal) {
  return Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", device_name, ":", device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      absl::StrCat("device: ", device_name, " device"));
}

}  // namespace

XlaDevice::Metadata::Metadata(
    int device_ordinal, se::Platform* platform, const DeviceType& device_type,
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns,
    PaddedShapeFn padded_shape_fn, bool use_multiple_streams)
    : device_ordinal_(device_ordinal),
      device_type_(device_type),
      platform_(platform),
      shape_determination_fns_(std::move(shape_determination_fns)),
      padded_shape_fn_(std::move(padded_shape_fn)),
      use_multiple_streams_(use_multiple_streams) {}

int XlaDevice::Metadata::device_ordinal() const { return device_ordinal_; }

se::Platform* XlaDevice::Metadata::platform() const { return platform_; }

xla::LocalClient* XlaDevice::Metadata::client() const {
  auto client = xla::ClientLibrary::GetOrCreateLocalClient(platform_);
  return client.value();
}

const DeviceType& XlaDevice::Metadata::jit_device_type() const {
  return device_type_;
}

/*static*/ absl::Status XlaDevice::GetMetadataFromDevice(
    DeviceBase* device, const XlaDevice::Metadata** metadata) {
  *metadata = nullptr;
  XlaDevice* xla_device = dynamic_cast<XlaDevice*>(device->UnderlyingDevice());
  if (xla_device == nullptr) {
    return absl::InternalError(absl::StrCat(
        "Cannot get XLA metadata from non-XLA device \"", device->name(),
        "\". GetMetadata must only be called on an XLA device. Either an "
        "internal bug has been triggered, or an XLA-specific op has been "
        "placed on the wrong device."));
  }
  *metadata = &(xla_device->xla_metadata_);
  return absl::OkStatus();
}

/* static */ absl::Status XlaDevice::GetMetadata(OpKernelContext* ctx,
                                                 const Metadata** metadata) {
  return GetMetadataFromDevice(ctx->device(), metadata);
}

/* static */ absl::Status XlaDevice::GetMetadata(OpKernelConstruction* ctx,
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
                    options.shape_determination_fns,
                    options.padded_shape_fn ? options.padded_shape_fn
                                            : DefaultPaddedShapeFn,
                    options.use_multiple_streams),
      device_ordinal_(options.device_ordinal),
      device_name_(options.device_name),
      jit_device_name_(options.compilation_device_name),
      platform_(options.platform),
      intra_op_parallelism_threads_(
          session_options.config.intra_op_parallelism_threads()),
      shape_determination_fns_(options.shape_determination_fns),
      allowed_devices_(options.allowed_devices) {
  if (options.shape_determination_fns.empty()) {
    LOG(ERROR) << "shape_representation_fns must be non-empty.";
  }
  VLOG(1) << "Created XLA device " << options.compilation_device_name << " "
          << options.device_ordinal << " " << this;
  thread_pool_.reset(new thread::ThreadPool(session_options.env, "xla_device",
                                            /*num_threads=*/1));
}

XlaDevice::~XlaDevice() {
  VLOG(1) << "Destroying XLA device " << jit_device_name_ << " " << this;
  mutex_lock lock(mu_);
  for (const auto& iter : device_contexts_) {
    iter->Unref();
  }
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
    VLOG(1) << "XlaDevice " << this << " uses AsyncValueAllocator";
    pjrt_allocator_ = std::make_unique<AsyncValueAllocator>();
    xla_allocator_ = pjrt_allocator_.get();
  }
  return xla_allocator_;
}

absl::Status XlaDevice::EnsureDeviceContextOk() {
  mutex_lock lock(mu_);
  return GetDeviceContextLocked().status();
}

absl::StatusOr<std::vector<DeviceContext*>>
XlaDevice::GetDeviceContextLocked() {
  if (device_contexts_.empty()) {
    for (const auto& iter : shape_determination_fns_) {
      auto device_context = new PjRtDeviceContext(iter);
      VLOG(1) << "XlaDevice " << this << " new PjRtDeviceContext "
              << device_context;
      device_contexts_.emplace_back(device_context);
    }
    if (use_accelerator_device_info_) {
      auto accelerator_device_info =
          std::make_unique<DeviceBase::AcceleratorDeviceInfo>();
      accelerator_device_info->default_context = device_contexts_.at(0);
      set_tensorflow_accelerator_device_info(accelerator_device_info.get());
      accelerator_device_info_ = std::move(accelerator_device_info);
      VLOG(1) << "XlaDevice " << this << " new AcceleratorDeviceInfo "
              << accelerator_device_info_.get();
    }
  }

  return device_contexts_;
}

absl::StatusOr<DeviceContext*> XlaDevice::GetDeviceContextWithIndex(int index) {
  mutex_lock lock(mu_);
  TF_ASSIGN_OR_RETURN(auto device_contexts, GetDeviceContextLocked());
  return device_contexts.at(index);
}

absl::StatusOr<DeviceContext*> XlaDevice::GetDeviceContextDefault() {
  return GetDeviceContextWithIndex(0);
}

absl::Status XlaDevice::UseAcceleratorDeviceInfo() {
  mutex_lock lock(mu_);
  use_accelerator_device_info_ = true;
  return GetDeviceContextLocked().status();
}

absl::Status XlaDevice::TryGetDeviceContext(DeviceContext** out_context) {
  TF_ASSIGN_OR_RETURN(auto device_context, GetDeviceContextDefault());
  device_context->Ref();
  *out_context = device_context;
  return absl::OkStatus();
}

// Warn about XLA_CPU/XLA_GPU exactly once.
static void ShowXlaDeviceDeprecationWarning(
    absl::string_view compilation_device_name) {
  static absl::once_flag once;
  if (absl::StrContains(compilation_device_name, "CPU") ||
      absl::StrContains(compilation_device_name, "GPU")) {
    absl::call_once(once, [] {
      LOG(INFO) << "XLA_GPU and XLA_CPU devices are deprecated and will be "
                   "removed in subsequent releases. Instead, use either "
                   "@tf.function(jit_compile=True) for must-compile "
                   "semantics, or run with TF_XLA_FLAGS=--tf_xla_auto_jit=2 "
                   "for auto-clustering best-effort compilation.";
    });
  }
}

void XlaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  VLOG(2) << "XlaDevice::Compute " << op_kernel->name() << ":"
          << op_kernel->type_string();
  ShowXlaDeviceDeprecationWarning(jit_device_name_.type_string());
  op_kernel->Compute(context);
}

void XlaDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                             AsyncOpKernel::DoneCallback done) {
  ShowXlaDeviceDeprecationWarning(jit_device_name_.type_string());
  VLOG(2) << "XlaDevice::ComputeAsync " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->ComputeAsync(context, done);
}

absl::Status XlaDevice::Sync() {
  VLOG(1) << "XlaDevice::Sync";
  return absl::OkStatus();
}

absl::Status XlaDevice::MakeTensorFromProto(
    DeviceContext* device_context, const TensorProto& tensor_proto,
    const AllocatorAttributes alloc_attrs, Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot parse tensor from proto: ", tensor_proto.DebugString()));
  }

  absl::Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Allocator* allocator;
    {
      mutex_lock lock(mu_);
      allocator = GetAllocatorLocked(alloc_attrs);
    }
    Tensor copy(allocator, parsed.dtype(), parsed.shape());
    TF_RETURN_IF_ERROR(
        device_context->CopyCPUTensorToDeviceSync(&parsed, this, &copy));
    *tensor = copy;
  }
  VLOG(2) << "Allocated tensor at " << DMAHelper::base(tensor);
  return status;
}

absl::Status XlaDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  VLOG(1) << "XlaDevice::MakeTensorFromProto";
  DeviceContext* device_context;
  TF_ASSIGN_OR_RETURN(device_context, GetDeviceContextDefault());
  return MakeTensorFromProto(device_context, tensor_proto, alloc_attrs, tensor);
}

void XlaDevice::SetAllowsSyncOnCompletion(bool sync_on_completion) {
  mutex_lock lock(mu_);
  sync_on_completion_ = sync_on_completion;
}

bool XlaDevice::AllowsSyncOnCompletion() const {
  mutex_lock lock(mu_);
  return sync_on_completion_;
}

void XlaDevice::SetHandleDeviceErrorCallback(
    std::function<absl::Status()> callback) {
  mutex_lock lock(mu_);
  device_error_callback_ = callback;
}

absl::Status XlaDevice::HandleDeviceError() {
  std::function<absl::Status()> local_device_error_callback;
  {
    mutex_lock lock(mu_);
    local_device_error_callback = device_error_callback_;
  }
  if (local_device_error_callback != nullptr) {
    return local_device_error_callback();
  }
  return absl::OkStatus();
}

absl::Status XlaDevice::RefreshStatus() { return absl::OkStatus(); }

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(
    const char* device, const char* jit_device,
    OpKernel* (*factory)(OpKernelConstruction*),
    absl::string_view kernel_class_name) {
  XlaOpRegistry::RegisterCompilationKernels();
  XlaDeviceOpRegistrations* registrations = new XlaDeviceOpRegistrations;
  for (const KernelDef* jit_def : XlaOpRegistry::DeviceKernels(
           jit_device,
           /*include_compilation_only_kernels=*/false)) {
    KernelDef* def = new KernelDef(*jit_def);
    const std::unordered_set<std::string>* constant_inputs =
        XlaOpRegistry::CompileTimeConstantInputArgNames(def->op());

    for (const std::string& arg_name : *constant_inputs) {
      def->add_host_memory_arg(arg_name);
    }

    def->set_device_type(device);
    registrations->op_kernel_registrars.emplace_back(
        new kernel_factory::OpKernelRegistrar(def, kernel_class_name, factory));
  }
  return registrations;
}

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device) {
  // Any op assigned to the device that isn't rewritten by the graph rewriter
  // gets executed by an XlaCompileOnDemandOp, which compiles it and executes
  // it just-in-time.
  auto factory = [](OpKernelConstruction* context) -> OpKernel* {
    return new XlaCompileOnDemandOp(context);
  };
  return RegisterXlaDeviceKernels(device, jit_device, factory,
                                  /*kernel_class_name=*/"XlaCompileOnDemandOp");
}

}  // namespace tensorflow
