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

// Registers the XLA_CPU device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "Host" (CPU) backend.

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class XlaCpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaCpuDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
  registration.autoclustering_policy =
      compile_on_demand
          ? XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested
          : XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.compile_resource_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_CPU, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_CPU, DEVICE_CPU_XLA_JIT);
  (void)registrations;

  TF_ASSIGN_OR_RETURN(auto platform,
                      se::MultiPlatformManager::PlatformWithName("Host"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_CPU;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_CPU_XLA_JIT;
  options.use_multiple_streams = false;
  devices->push_back(absl::make_unique<XlaDevice>(session_options, options));
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_CPU, XlaCpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 12> kAllXlaCpuTypes = {
    {DT_UINT8, DT_QUINT8, DT_INT8, DT_QINT8, DT_INT32, DT_QINT32, DT_INT64,
     DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BOOL}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_CPU, XlaLocalLaunchOp, kAllXlaCpuTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_CPU, XlaCompileOp, kAllXlaCpuTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_CPU, XlaRunOp, kAllXlaCpuTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_CPU, kAllXlaCpuTypes);

}  // namespace tensorflow
