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

#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"
#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

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
  IpuDevice(const SessionOptions& options, const XlaDevice::Options& devopts)
      : XlaDevice(options, devopts),
        ordinal_(devopts.device_ordinal),
        poplar_platform_(static_cast<xp::PoplarPlatform*>(devopts.platform)) {}

  Status Init(const tensorflow::IPUOptions& options) {
    UseGpuDeviceInfo();
    return poplar_platform_->ConfigurePoplarDevice(ordinal_, options);
  }

  virtual ~IpuDevice() {}

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

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_IPU_XLA_JIT;
  registration.requires_compilation = true;
  registration.enable_jit_by_default = false;
  registration.compile_resource_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_IPU, registration);

  int config_count = options.config.ipu_options().device_config_size();

  auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
  if (!platform.ok()) {
    return platform.status();
  }

  auto* p = static_cast<xp::PoplarPlatform*>(platform.ValueOrDie());

  XlaDevice::Options devopts;
  devopts.platform = platform.ValueOrDie();
  devopts.device_name_prefix = name_prefix;
  devopts.compilation_device_name = DEVICE_IPU_XLA_JIT;
  devopts.device_name = DEVICE_XLA_IPU;

  int num_devices = p->VisibleDeviceCount();
  num_devices = std::min(num_devices, config_count);
  num_devices = std::max(num_devices, 1);

  for (int ordinal = 0; ordinal < num_devices; ordinal++) {
    devopts.device_ordinal = ordinal;

    auto* device = new IpuDevice(options, devopts);
    TF_RETURN_IF_ERROR(device->Init(options.config.ipu_options()));
    devices->push_back(device);
  }

  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaLocalLaunchOp, kIpuAllTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_XLA_IPU, XlaCompileOp, kIpuAllTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_XLA_IPU, XlaRunOp, kIpuAllTypes);

REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, kIpuAllTypes);

// Additional ops not explicitly defined by standard JIT
REGISTER_XLA_OP(Name("ArgMax")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("dimension"),
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
