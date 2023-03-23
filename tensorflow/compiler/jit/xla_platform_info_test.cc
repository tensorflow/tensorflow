/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_platform_info.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {
using XlaDeviceCompiler =
    DeviceCompiler<xla::LocalExecutable, xla::LocalClient>;
using XlaDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::LocalExecutable, xla::LocalClient>;

class XlaPlatformInfoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensorflow::GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;
  }

  DeviceSetup device_setup_;
};

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerXlaDeviceMetadata) {
  device_setup_.AddDevicesAndSetUp({DEVICE_XLA_GPU});

  Device* device = device_setup_.GetDevice(DEVICE_XLA_GPU);
  const XlaDevice::Metadata* metadata = nullptr;
  TF_CHECK_OK(XlaDevice::GetMetadataFromDevice(device, &metadata));
  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, device_setup_.flr(),
                                      platform_info, &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), metadata->jit_device_type());
  EXPECT_EQ(xla_device_compiler->client(), metadata->client());
}

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerTpuDevice) {
  DeviceType compilation_device_type = DeviceType(DEVICE_TPU_XLA_JIT);

  // Instead of creating/initializing a TPU device, create a dummy platform_info
  // and use a nullptr for Device for testing purposes. Only
  // XlaPlatformInfo::device_type() is needed to build the appropriate
  // XlaDeviceCompiler.
  Device* device = nullptr;
  XlaPlatformInfo platform_info(DeviceType(DEVICE_TPU), nullptr, nullptr,
                                nullptr);

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, nullptr, platform_info,
                                      &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), compilation_device_type);
  // TFRT-TPU is used if device type is `DEVICE_TPU` and `platform_info` does
  // not have `xla_device_metadata`. XlaDeviceCompiler/xla::LocalClient is not
  // used in this case.
  EXPECT_EQ(xla_device_compiler->client(), nullptr);
}

TEST_F(XlaPlatformInfoTest, BuildXlaDeviceCompilerRegularDevice) {
  device_setup_.AddDevicesAndSetUp({DEVICE_GPU});
  Device* device = device_setup_.GetDevice(DEVICE_GPU);

  XlaPlatformInfo platform_info = XlaPlatformInfoFromDevice(device);

  XlaDeviceCompiler* xla_device_compiler = nullptr;
  TF_EXPECT_OK(BuildXlaDeviceCompiler(device, device_setup_.flr(),
                                      platform_info, &xla_device_compiler));
  core::ScopedUnref xla_device_compiler_ref(xla_device_compiler);

  EXPECT_EQ(xla_device_compiler->device_type(), DeviceType(DEVICE_GPU_XLA_JIT));
  EXPECT_TRUE(xla_device_compiler->client() != nullptr);
}

}  // namespace
}  // namespace tensorflow
