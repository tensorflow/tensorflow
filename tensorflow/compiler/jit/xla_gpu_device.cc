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

#include <set>
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class XlaGpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaGpuDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_GPU_XLA_JIT;
  registration.autoclustering_policy =
      XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.compile_resource_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_GPU, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_GPU, DEVICE_GPU_XLA_JIT);
  (void)registrations;

  auto platform = se::MultiPlatformManager::PlatformWithName("CUDA");
  if (!platform.ok()) {
    // Treat failures as non-fatal; there might not be a GPU in the machine.
    VLOG(1) << "Failed to create XLA_GPU device: " << platform.status();
    return Status::OK();
  }
  string allowed_gpus =
      session_options.config.gpu_options().visible_device_list();
  std::set<int> gpu_ids;
  int num_visible_devices = platform.ValueOrDie()->VisibleDeviceCount();
  if (allowed_gpus.empty()) {
    for (int i = 0; i < num_visible_devices; ++i) gpu_ids.insert(i);
  } else {
    // For loop below is copied from gpu/gpu_device.cc. It validates
    // configuration string. It should be redundant since code would fail there
    // before it gets to here.
    const std::vector<string> visible_devices =
        absl::StrSplit(allowed_gpus, ',');
    for (const string& platform_gpu_id_str : visible_devices) {
      int32 platform_gpu_id;
      if (!absl::SimpleAtoi(platform_gpu_id_str, &platform_gpu_id)) {
        return errors::InvalidArgument(
            "Could not parse entry in 'visible_device_list': '",
            platform_gpu_id_str, "'. visible_device_list = ", allowed_gpus);
      }
      if (platform_gpu_id < 0 || platform_gpu_id >= num_visible_devices) {
        return errors::InvalidArgument(
            "'visible_device_list' listed an invalid GPU id '", platform_gpu_id,
            "' but visible device count is ", num_visible_devices);
      }
      gpu_ids.insert(platform_gpu_id);
    }
  }
  for (const auto i : gpu_ids) {
    // Skip devices that are not in the set.
    XlaDevice::Options options;
    options.platform = platform.ValueOrDie();
    options.device_name_prefix = name_prefix;
    options.device_name = DEVICE_XLA_GPU;
    options.device_ordinal = i;
    options.compilation_device_name = DEVICE_GPU_XLA_JIT;
    options.use_multiple_streams = true;
    auto device = absl::make_unique<XlaDevice>(session_options, options);

    Status status = device->UseGpuDeviceInfo();
    if (!status.ok()) {
      errors::AppendToMessage(&status, "while setting up ", DEVICE_GPU_XLA_JIT,
                              " device number ", i);
      return status;
    }

    devices->push_back(std::move(device));
  }
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_GPU, XlaGpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 13> kAllXlaGpuTypes = {
    {DT_UINT8, DT_QUINT8, DT_INT8, DT_QINT8, DT_INT32, DT_QINT32, DT_INT64,
     DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BOOL, DT_BFLOAT16}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_GPU, XlaLocalLaunchOp, kAllXlaGpuTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_GPU, XlaCompileOp, kAllXlaGpuTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_GPU, XlaRunOp, kAllXlaGpuTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_GPU, kAllXlaGpuTypes);

}  // namespace tensorflow
