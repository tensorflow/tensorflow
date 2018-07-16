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

// Registers the XLA_INTERPRETER device which exposes the XLA Interpreter.

#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {

const char* const DEVICE_XLA_INTERPRETER = "XLA_INTERPRETER";
const char* const DEVICE_INTERPRETER_XLA_JIT = "XLA_INTERPRETER_JIT";

constexpr std::array<DataType, 6> kExecAllTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BOOL}};

class XlaInterpreterDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaInterpreterDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(
      DEVICE_XLA_INTERPRETER, DEVICE_INTERPRETER_XLA_JIT);
  (void)registrations;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_INTERPRETER_XLA_JIT;
  registration.requires_compilation = true;
  registration.enable_jit_by_default = false;
  registration.compile_resource_ops = true;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Interpreter", DEVICE_XLA_INTERPRETER, 0,
                                       DEVICE_INTERPRETER_XLA_JIT, options,
                                       name_prefix, registration,
                                       /*transfer_as_literal=*/false,
                                       /*use_multiple_streams=*/false,
                                       /*shape_representation_fn=*/{},
                                       /*padded_shape_fn=*/{}, &device));
  devices->push_back(device.release());
  return Status::OK();
}

// Set priority to be below the default priority (50), so that Interpreter is
// not selected as a high priority device over other default devices. See
// constructor comments for Registrar in
// tensorflow/core/common_runtime/device_factory.h for a list of priority for
// devices.
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_INTERPRETER,
                              XlaInterpreterDeviceFactory, 40);

// Kernel registrations
static bool OpFilter(KernelDef* kdef) { return true; }

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_INTERPRETER, XlaLocalLaunchOp,
                           kExecAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_INTERPRETER, kExecAllTypes);
REGISTER_XLA_BACKEND(DEVICE_INTERPRETER_XLA_JIT, kExecAllTypes, OpFilter);

}  // namespace tensorflow
