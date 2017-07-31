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

#include "tensorflow/compiler/jit/kernels/xla_device_launch_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {

const char* const DEVICE_XLA_EXEC = "XLA_EXEC";
const char* const DEVICE_EXEC_XLA_JIT = "XLA_EXEC_JIT";

constexpr std::array<DataType, 5> kExecAllTypes = {
    {DT_INT32, DT_FLOAT, DT_BOOL, DT_DOUBLE, DT_INT64}};

class XlaExaDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaExaDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_EXEC, DEVICE_EXEC_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Executor", DEVICE_XLA_EXEC, 0,
                                       DEVICE_EXEC_XLA_JIT, options,
                                       name_prefix, &device));
  devices->push_back(device.release());
  return Status::OK();
}

// Set priority to be below the default priority (50), so that Executor is not
// selected as a high priority device over other default devices.
// See constructor comments for Registrar in
// tensorflow/core/common_runtime/device_factory.h for a list of priority for
// devices.
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_EXEC, XlaExaDeviceFactory, 40);

// Kernel registrations

static bool OpFilter(KernelDef* kdef) { return true; }

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_EXEC, XlaDeviceLaunchOp, kExecAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_EXEC, kExecAllTypes);
REGISTER_XLA_BACKEND(DEVICE_EXEC_XLA_JIT, kExecAllTypes, OpFilter);

}  // namespace tensorflow
