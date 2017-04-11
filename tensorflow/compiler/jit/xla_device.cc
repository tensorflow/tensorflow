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

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/stream_executor_util.h"

namespace tensorflow {

/* static */ Status XlaDevice::Create(
    const string& platform_name, const string& device_name, int device_ordinal,
    const string& jit_device_name, const SessionOptions& options,
    const string& name_prefix, std::unique_ptr<XlaDevice>* device) {
  VLOG(1) << "XlaDevice::Create " << platform_name << " " << device_name << ":"
          << device_ordinal;

  // These are no-ops if they have already been done previously for
  // this device_name/compilation_device_name pair.
  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = jit_device_name;
  registration.requires_compilation = true;
  registration.enable_jit_by_default = false;
  registration.compile_resource_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(device_name, registration);

  auto platform = perftools::gputools::MultiPlatformManager::PlatformWithName(
      platform_name);
  if (!platform.ok()) {
    return StreamExecutorUtil::ConvertStatus(platform.status());
  }

  const DeviceAttributes attrs = Device::BuildDeviceAttributes(
      strings::StrCat(name_prefix, "/device:", device_name, ":",
                      device_ordinal),
      DeviceType(device_name), Bytes(16ULL << 30), DeviceLocality(),
      strings::StrCat("device: ", device_name, " device"));

  static Allocator* allocator = new XlaDeviceAllocator;
  device->reset(new XlaDevice(options, attrs, device_ordinal,
                              DeviceType(jit_device_name),
                              platform.ValueOrDie(), allocator));
  return Status::OK();
}

XlaDevice::Metadata::Metadata(int device_ordinal,
                              perftools::gputools::Platform* platform,
                              const DeviceType& device_type)
    : device_ordinal_(device_ordinal),
      device_type_(device_type),
      platform_(platform) {}

int XlaDevice::Metadata::device_ordinal() const { return device_ordinal_; }

perftools::gputools::Platform* XlaDevice::Metadata::platform() const {
  return platform_;
}

XlaDevice::Metadata::~Metadata() {}

xla::LocalClient* XlaDevice::Metadata::client() const {
  auto client = xla::ClientLibrary::GetOrCreateLocalClient(platform_);
  return client.ValueOrDie();
}

const DeviceType& XlaDevice::Metadata::jit_device_type() const {
  return device_type_;
}

string XlaDevice::Metadata::DebugString() { return "XLA device metadata"; }

XlaDevice::XlaDevice(const SessionOptions& options,
                     const DeviceAttributes& attrs, int device_ordinal,
                     const DeviceType& jit_device_name,
                     perftools::gputools::Platform* platform,
                     Allocator* xla_allocator)
    : LocalDevice(options, attrs, xla_allocator),
      device_ordinal_(device_ordinal),
      jit_device_name_(jit_device_name),
      xla_allocator_(xla_allocator),
      platform_(platform) {
  // Store the platform in the resource manager so Ops can retrieve it
  // e.g., to lazily create a XlaCompilationCache object.
  TF_CHECK_OK(resource_manager()->Create<Metadata>(
      resource_manager()->default_container(), "xla_metadata",
      new Metadata(device_ordinal_, platform_, jit_device_name_)));
}
XlaDevice::~XlaDevice() {}

xla::LocalClient* XlaDevice::client() const {
  // We lazily create the client because the platform commits to the
  // details of the host hardware when the client is created, so we
  // don't want to do it until we get a chance to hook the platform up
  // to a simulator.

  // For now GetOrCreateLocalClient always returns success when passed
  // a non-null platform. If that changes we may have to plumb in some
  // way to pass Status back.
  return xla::ClientLibrary::GetOrCreateLocalClient(platform_).ValueOrDie();
}

Allocator* XlaDevice::GetAllocator(AllocatorAttributes attr) {
  if (attr.on_host()) {
    return cpu_allocator();
  } else {
    return xla_allocator_;
  }
}

Status XlaDevice::FillContextMap(const Graph* graph,
                                 DeviceContextMap* device_context_map) {
  VLOG(1) << "XlaDevice::FillContextMap";
  device_context_map->resize(graph->num_node_ids());
  XlaDeviceContext* ctx = new XlaDeviceContext(client());
  for (Node* n : graph->nodes()) {
    VLOG(2) << n->id() << " : " << n->type_string() << " : " << n->name();
    ctx->Ref();
    (*device_context_map)[n->id()] = ctx;
  }
  ctx->Unref();
  return Status::OK();
}

void XlaDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  VLOG(1) << "XlaDevice::Compute " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->Compute(context);
}

void XlaDevice::ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                             AsyncOpKernel::DoneCallback done) {
  VLOG(1) << "XlaDevice::ComputeAsync " << op_kernel->name() << ":"
          << op_kernel->type_string();
  op_kernel->ComputeAsync(context, done);
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
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());
    Notification n;
    XlaTransferManager manager(client());
    manager.CopyCPUTensorToDevice(&parsed, this, &copy,
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

XlaDeviceOpRegistrations* RegisterXlaDeviceKernels(const char* device,
                                                   const char* jit_device) {
  XlaOpRegistry::RegisterCompilationKernels();
  XlaDeviceOpRegistrations* registrations = new XlaDeviceOpRegistrations;
  auto dummy_factory = [](OpKernelConstruction* context) -> OpKernel* {
    return new XlaDeviceDummyOp(context);
  };
  for (const KernelDef* jit_def : XlaOpRegistry::DeviceKernels(jit_device)) {
    KernelDef* def = new KernelDef(*jit_def);
    def->set_device_type(device);
    registrations->op_kernel_registrars.emplace_back(
        new kernel_factory::OpKernelRegistrar(def, "XlaDeviceDummyOp",
                                              dummy_factory));
  }
  return registrations;
}

}  // namespace tensorflow
