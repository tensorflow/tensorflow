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

// Registers the XLA_GPU device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "CUDA" (GPU) backend.

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

const char* const DEVICE_XLA_GPU = "XLA_GPU";

class XlaGpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaGpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_GPU, DEVICE_GPU_XLA_JIT);
  (void)registrations;

  std::unique_ptr<XlaDevice> device;
  Status status =
      XlaDevice::Create("CUDA", DEVICE_XLA_GPU, 0, DEVICE_GPU_XLA_JIT, options,
                        name_prefix, &device);
  if (!status.ok()) {
    // Treat failures as non-fatal; there might not be a GPU in the machine.
    VLOG(1) << "Failed to create XLA_GPU device: " << status;
    return Status::OK();
  }
  devices->push_back(device.release());
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_GPU, XlaGpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 5> kAllXlaGpuTypes = {
    {DT_INT32, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_GPU, XlaDeviceLaunchOp, kAllXlaGpuTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_GPU, kAllXlaGpuTypes);

}  // namespace tensorflow
