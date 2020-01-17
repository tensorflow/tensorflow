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
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Returns a set containing the device ids contained in visible_device_list or
// nullopt if it is empty. It returns error in case of malformed configuration
// string.
static xla::StatusOr<absl::optional<std::set<int>>> ParseVisibleDeviceList(
    const string& visible_device_list) {
  std::set<int> gpu_ids;
  if (visible_device_list.empty()) {
    return {{absl::nullopt}};
  }
  const std::vector<string> visible_devices =
      absl::StrSplit(visible_device_list, ',');
  for (const string& platform_gpu_id_str : visible_devices) {
    int32 platform_gpu_id;
    if (!absl::SimpleAtoi(platform_gpu_id_str, &platform_gpu_id)) {
      return errors::InvalidArgument(
          "Could not parse entry in 'visible_device_list': '",
          platform_gpu_id_str,
          "'. visible_device_list = ", visible_device_list);
    }
    gpu_ids.insert(platform_gpu_id);
  }
  return {{gpu_ids}};
}

class XlaGpuDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status XlaGpuDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    LOG(INFO) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return Status::OK();
  }

  auto platform = se::MultiPlatformManager::PlatformWithName("CUDA");
  if (!platform.ok()) {
    // Treat failures as non-fatal; there might not be a GPU in the machine.
    VLOG(1) << "Failed to create XLA_GPU device: " << platform.status();
    return Status::OK();
  }

  int device_count = platform.ValueOrDie()->VisibleDeviceCount();
  if (device_count <= 0) {
    return Status::OK();
  }

  for (int i = 0; i < device_count; ++i) {
    devices->push_back(
        absl::StrCat("/physical_device:", DEVICE_XLA_GPU, ":", i));
  }

  return Status::OK();
}

Status XlaGpuDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  XlaDeviceFlags* flags = GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    LOG(INFO) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return Status::OK();
  }

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_GPU_XLA_JIT;
  registration.autoclustering_policy =
      XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
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

  auto iter = session_options.config.device_count().find("GPU");
  if (iter != session_options.config.device_count().end() &&
      iter->second == 0) {
    // Device count for GPU is 0.
    return Status::OK();
  }

  string allowed_gpus =
      session_options.config.gpu_options().visible_device_list();
  absl::optional<std::set<int>> gpu_ids =
      ParseVisibleDeviceList(allowed_gpus).ValueOrDie();
  if (!gpu_ids) {
    gpu_ids.emplace();
    // Fill the gpu_ids set with all devices if config string is empty.
    for (int i = 0; i < platform.ValueOrDie()->VisibleDeviceCount(); ++i) {
      gpu_ids->insert(i);
    }
  }
  for (int i : *gpu_ids) {
    XlaDevice::Options options;
    options.platform = platform.ValueOrDie();
    options.device_name_prefix = name_prefix;
    options.device_name = DEVICE_XLA_GPU;
    options.device_ordinal = i;
    options.compilation_device_name = DEVICE_GPU_XLA_JIT;
    options.use_multiple_streams = true;
    options.allowed_devices = gpu_ids;
    auto device = absl::make_unique<XlaDevice>(session_options, options);

    Status status = device->UseGpuDeviceInfo();
    if (!status.ok()) {
      LOG(INFO) << "Ignoring visible " << DEVICE_GPU_XLA_JIT
                << " device. Device number is " << i << ", reason: " << status;
      continue;
    }

    devices->push_back(std::move(device));
  }
  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_GPU, XlaGpuDeviceFactory);

// Kernel registrations

constexpr std::array<DataType, 16> kAllXlaGpuTypes = {
    {DT_UINT8, DT_QUINT8, DT_UINT16, DT_INT8, DT_QINT8, DT_INT16, DT_INT32,
     DT_QINT32, DT_INT64, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
     DT_COMPLEX128, DT_BOOL, DT_BFLOAT16}};

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_GPU, XlaLocalLaunchOp, kAllXlaGpuTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_GPU, XlaCompileOp, kAllXlaGpuTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_GPU, XlaRunOp, kAllXlaGpuTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_GPU, kAllXlaGpuTypes);

}  // namespace tensorflow
