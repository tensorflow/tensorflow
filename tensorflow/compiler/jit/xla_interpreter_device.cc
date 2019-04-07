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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {

const char* const DEVICE_XLA_INTERPRETER = "XLA_INTERPRETER";
const char* const DEVICE_INTERPRETER_XLA_JIT = "XLA_INTERPRETER_JIT";

constexpr std::array<DataType, 10> kExecAllTypes = {
    {DT_INT8, DT_INT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128, DT_BOOL, DT_BFLOAT16}};

class XlaInterpreterDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaInterpreterDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  static XlaDeviceOpRegistrations* registrations = RegisterXlaDeviceKernels(
      DEVICE_XLA_INTERPRETER, DEVICE_INTERPRETER_XLA_JIT);
  (void)registrations;

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_INTERPRETER_XLA_JIT;
  registration.autoclustering_policy =
      XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_INTERPRETER,
                                           registration);

  TF_ASSIGN_OR_RETURN(
      auto platform, se::MultiPlatformManager::PlatformWithName("Interpreter"));

  XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_INTERPRETER;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_INTERPRETER_XLA_JIT;
  options.use_multiple_streams = false;
  devices->push_back(absl::make_unique<XlaDevice>(session_options, options));

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
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_INTERPRETER, XlaCompileOp,
                            kExecAllTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_INTERPRETER, XlaRunOp, kExecAllTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_INTERPRETER, kExecAllTypes);
REGISTER_XLA_BACKEND(DEVICE_INTERPRETER_XLA_JIT, kExecAllTypes, OpFilter);

}  // namespace tensorflow
