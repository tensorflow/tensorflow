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

#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/jit/legacy_flags/xla_device_flags.h"
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
                       std::vector<Device*>* devices) override;
};

Status XlaCpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  legacy_flags::XlaDeviceFlags* flags = legacy_flags::GetXlaDeviceFlags();
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_CPU_XLA_JIT;
  registration.requires_compilation = !compile_on_demand;
  registration.enable_jit_by_default = false;
  registration.compile_resource_ops = true;

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_CPU, DEVICE_CPU_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Host", DEVICE_XLA_CPU, 0,
                                       DEVICE_CPU_XLA_JIT, options, name_prefix,
                                       registration,
                                       /*transfer_as_literal=*/false,
                                       /*shape_representation_fn=*/{},
                                       /*padded_shape_fn=*/{}, &device));
  devices->push_back(device.release());
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_CPU, XlaCpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 7> kAllXlaCpuTypes = {
    {DT_INT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BOOL}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_CPU, XlaLocalLaunchOp, kAllXlaCpuTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_CPU, kAllXlaCpuTypes);

}  // namespace tensorflow
