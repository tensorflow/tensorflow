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

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

#include <poplar/Engine.hpp>
#include <popnn/Net.hpp>

namespace tensorflow {

const char* const DEVICE_XLA_IPU = "XLA_IPU";
const char* const DEVICE_IPU_XLA_JIT = "XLA_CPU_JIT";

class XlaIpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaIpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_IPU, DEVICE_IPU_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  TF_RETURN_IF_ERROR(XlaDevice::Create("Host", DEVICE_XLA_IPU, 0,
                                       DEVICE_IPU_XLA_JIT, options, name_prefix,
                                       &device));
  devices->push_back(device.release());
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 5> kAllXlaIpuTypes = {
    {DT_INT32, DT_FLOAT, DT_HALF, DT_BOOL}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaDeviceLaunchOp, kAllXlaIpuTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, kAllXlaIpuTypes);

}  // namespace tensorflow
