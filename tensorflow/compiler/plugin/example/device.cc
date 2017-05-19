/* Copyright 2017 Graphcore Ltd
 */

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

const char* const DEVICE_XLA_EXA = "XLA_EXA";
const char* const DEVICE_EXA_XLA_JIT = "XLA_EXA_JIT";

constexpr std::array<DataType, 5> kExaAllTypes = {
    {DT_INT32, DT_FLOAT, DT_BOOL}};

class XlaExaDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaExaDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_EXA, DEVICE_EXA_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Example", DEVICE_XLA_EXA, 0,
                                       DEVICE_EXA_XLA_JIT, options, name_prefix,
                                       &device));
  devices->push_back(device.release());
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_EXA, XlaExaDeviceFactory, 210);

// Kernel registrations

static bool OpFilter(KernelDef* kdef) { return true; }

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_EXA, XlaDeviceLaunchOp, kExaAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_EXA, kExaAllTypes);
REGISTER_XLA_BACKEND(DEVICE_EXA_XLA_JIT, kExaAllTypes, OpFilter);

}  // namespace tensorflow
