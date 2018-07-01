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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"

#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/kernels/no_op.h"

namespace xp = ::xla::poplarplugin;

namespace tensorflow {

namespace {

Status DefaultPaddedShapeFn(const Tensor& tensor, xla::Shape* shape) {
  const tensorflow::XlaTensor* xla_tensor =
      tensorflow::XlaTensor::FromTensor(&tensor);
  if (xla_tensor == nullptr) {
    return TensorShapeToXLAShape(tensor.dtype(), tensor.shape(), shape);
  }

  const xla::ShapedBuffer& shaped_buffer = xla_tensor->shaped_buffer();
  *shape = shaped_buffer.on_device_shape();
  return Status::OK();
}

}  // namespace

class IpuDevice : public XlaDevice {
 public:
  IpuDevice(const SessionOptions& options, const DeviceAttributes& attrs,
            int device_ordinal, const DeviceType& jit_device_name,
            se::Platform* platform, bool transfer_as_literal)
      : XlaDevice(options, attrs, device_ordinal, jit_device_name, platform,
                  transfer_as_literal, {}, DefaultPaddedShapeFn),
        ordinal_(device_ordinal),
        poplar_platform_(static_cast<xp::PoplarPlatform*>(platform)) {}

  Status Init(const tensorflow::IPUOptions& options) {
    return poplar_platform_->ConfigurePoplarDevices(this, ordinal_, options);
  }

  virtual ~IpuDevice() { poplar_platform_->ClosePoplarDevice(this, ordinal_); }

 private:
  int ordinal_;
  xp::PoplarPlatform* poplar_platform_;
};

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

  auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
  if (!platform.ok()) {
    return platform.status();
  }

  auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  int visible_devices = p->VisibleDeviceCount();
  for (int ordinal = 0; ordinal < visible_devices; ordinal++) {
    XlaOpRegistry::DeviceRegistration registration;
    registration.compilation_device_name = DEVICE_IPU_XLA_JIT;
    registration.requires_compilation = true;
    registration.enable_jit_by_default = false;
    registration.compile_resource_ops = true;
    XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_IPU, registration);

    se::StreamExecutor* executor;
    TF_ASSIGN_OR_RETURN(executor, p->ExecutorForDevice(ordinal));
    auto* e = static_cast<xp::PoplarExecutor*>(executor->implementation());
    auto& t = e->GetPoplarDevice().getTarget();

    int64 mem = t.getNumIPUs() * t.getTilesPerIPU() * t.getBytesPerTile();

    std::string target_type_name;
    switch (t.getTargetType()) {
      case poplar::TargetType::IPU:
        target_type_name = "IPU Device (IPU configuration)";
        break;
      case poplar::TargetType::IPU_MODEL:
        target_type_name = "IPU Device (IPU Model configuration)";
        break;
      case poplar::TargetType::CPU:
        target_type_name = "IPU Device (CPU configuration)";
        break;
      default:
        target_type_name = "IPU Device (Unknown configuration)";
        break;
    }

    const DeviceAttributes attrs = Device::BuildDeviceAttributes(
        strings::StrCat(name_prefix, "/device:IPU:", ordinal),
        DeviceType(DEVICE_XLA_IPU), Bytes(mem), DeviceLocality(),
        target_type_name);

    auto* device = new IpuDevice(options, attrs, ordinal,
                                 DeviceType(DEVICE_IPU_XLA_JIT), p, false);

    TF_RETURN_IF_ERROR(device->Init(options.config.ipu_options()));

    devices->push_back(device);
  }

  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaLocalLaunchOp, kIpuAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, kIpuAllTypes);

// Additional ops not explicitly defined by standard JIT
REGISTER_XLA_OP(Name("ArgMax")
.Device(DEVICE_IPU_XLA_JIT)
.CompileTimeConstInput("dimension"),
XlaArgMaxOp);

REGISTER_KERNEL_BUILDER(Name("RefEnter").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("RefExit").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("RefMerge").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("RefNextIteration").Device(DEVICE_IPU_XLA_JIT),
    NoOp);
REGISTER_KERNEL_BUILDER(Name("RefSwitch").Device(DEVICE_IPU_XLA_JIT), NoOp);

REGISTER_KERNEL_BUILDER(Name("Enter").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("Exit").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("LoopCond").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("Merge").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("NextIteration").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_KERNEL_BUILDER(Name("Switch").Device(DEVICE_IPU_XLA_JIT), NoOp);

}  // namespace tensorflow
