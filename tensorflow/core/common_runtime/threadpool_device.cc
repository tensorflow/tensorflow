/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   BusAdjacency bus_adjacency,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, bus_adjacency),
                  allocator),
      allocator_(allocator) {}

ThreadPoolDevice::~ThreadPoolDevice() {}

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
    op_kernel->Compute(context);
  }
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  *tensor = parsed;
  return Status::OK();
}

}  // namespace tensorflow

#include "tensorflow/core/platform/regexp.h"
namespace {
static RE2* tpd_valid_op_name_pattern = new RE2("[A-Za-z0-9.][A-Za-z0-9_.\\-/]*");
static RE2* valid_data_input_pattern =
    new RE2("[A-Za-z0-9.][A-Za-z0-9_.\\-/]*(\\:(0|([1-9][0-9]*)))?");
}

#include <sys/syslog.h>

class SomeFactory {
 public:
  SomeFactory() {
    syslog(LOG_ERR, "SomeFactory was called from threadpool_device!");
  }
};

SomeFactory g_some_factory;

__attribute__((constructor, section ("__TEXT,__text_no_strip,regular,no_dead_strip")))
void SomeFactoryInitFuncTPD() {
  syslog(LOG_ERR, "SomeFactoryInitFunc was called from threadpool_device!");
}

tensorflow::Status TPDValidateOpName(const tensorflow::string& op_name) {
  if (RE2::FullMatch(op_name, *tpd_valid_op_name_pattern)) {
    return tensorflow::Status::OK();
  } else {
    return tensorflow::Status::OK();
  }
}

volatile auto g_func_address = &SomeFactoryInitFuncTPD;
