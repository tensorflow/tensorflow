/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tfrt/cpp_tests/test_util.h""  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tfrt {
namespace {

using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::StartsWith;

TEST(UtilsTest, ConvertTfDeviceNameToTfrt) {
  const std::string device_name_prefix =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  tensorflow::StaticDeviceMgr device_mgr(
      tensorflow::DeviceFactory::NewDevice("CPU", {}, device_name_prefix));
  auto eager_context = new tensorflow::EagerContext(
      tensorflow::SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*async=*/false, &device_mgr, /*device_mgr_owned=*/false,
      /*rendezvous=*/nullptr);

  EXPECT_FALSE(ConvertTfDeviceNameToTfrt("unknown_device", eager_context));
  EXPECT_THAT(std::string(*ConvertTfDeviceNameToTfrt(device_name_prefix.c_str(),
                                                     eager_context)),
              StartsWith(device_name_prefix));

  eager_context->Unref();
}

TEST(UtilsTest, ConvertTfDTypeToTfrtDType) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE)                          \
  EXPECT_EQ(ConvertTfDTypeToTfrtDType(tensorflow::TF_DTYPE), \
            DType(DType::TFRT_DTYPE));
#include "tensorflow/core/tfrt/utils/dtype.def"  // NOLINT

  EXPECT_EQ(ConvertTfDTypeToTfrtDType(tensorflow::DT_HALF_REF), DType());
}

TEST(UtilsTest, CreateDummyTfDevices) {
  const std::vector<std::string> device_name{"/device:cpu:0", "/device:gpu:1"};
  std::vector<std::unique_ptr<tensorflow::Device>> dummy_tf_devices;

  CreateDummyTfDevices(device_name, &dummy_tf_devices);

  ASSERT_THAT(dummy_tf_devices, SizeIs(2));

  EXPECT_EQ(dummy_tf_devices[0]->name(), device_name[0]);
  EXPECT_EQ(dummy_tf_devices[0]->device_type(), tensorflow::DEVICE_TPU_SYSTEM);
  EXPECT_THAT(dummy_tf_devices[0]->attributes().physical_device_desc(),
              HasSubstr("device: TFRT TPU SYSTEM device"));
  EXPECT_EQ(dummy_tf_devices[1]->name(), device_name[1]);
}

TEST(UtilsTest, AddDummyTfrtDevices) {
  std::unique_ptr<HostContext> host_ctx = CreateHostContext();
  const std::vector<std::string> device_name{"/device:tpu:0"};
  AddDummyTfrtDevices(device_name, host_ctx.get());

  RCReference<Device> device0 =
      host_ctx->GetDeviceManager()->GetDeviceRef<Device>(device_name[0]);
  ASSERT_TRUE(device0);
  EXPECT_EQ(device0->name(), device_name[0]);

  RCReference<Device> device1 =
      host_ctx->GetDeviceManager()->GetDeviceRef<Device>("no-such-device");
  EXPECT_FALSE(device1);
}

}  // namespace
}  // namespace tfrt
